## Lending Club's prepayment curves are all percentages of face value by month, 
## rather than percentage of outstanding face value, which we do.  Should we change?
import os
import json
import numpy as np
import pandas as pd
import scipy
import loanstats
from constants import PathManager as paths
from collections import Counter
from matplotlib import pyplot as plt


class CashFlowModel(object):
    required_fields = ['term', 'intRate', 'default_risk', 'prepay_risk']
    income_tax_rate = 0.5
    write_off_tax_shield = 0.0  #0.2

    def __init__(self, model_dir=None):
        self.model_dir = model_dir
        self.default_curves = DefaultCurve(self.model_dir)
        self.prepay_curves = PrepayCurve(self.model_dir)

    def process(self, loan_data):
        inputs = [loan_data[field] for field in self.required_fields]
        irr, irr_after_tax = self.calc_irr(*inputs)
        loan_data['irr'] = irr
        loan_data['irr_after_tax'] = irr_after_tax

    def calc_irr(self, term, int_rate, default_risk, prepay_risk):
        ''' In the code below, think of the calculations as applying to a pool of 100 loans, 
        with a percentage fully prepaying each month and a percentage defaulting each month.'''

        net_payment_pct = 0.99  #LC charges 1% fee on all incoming payments

        monthly_int_rate = int_rate/1200.
        pmt = np.pmt(monthly_int_rate, term, -1)
        periods = 1 + np.arange(term)
        ipmt = np.ipmt(monthly_int_rate, periods, term, -1)
        ppmt = np.ppmt(monthly_int_rate, periods, term, -1)

        default_curve = self.default_curves.custom_curve(term, default_risk)
        prepay_curve = self.prepay_curves.custom_curve(term, prepay_risk)

        # Adjust prepay curve so that prepay[11] is the odds the loan is prepaid any time
        # before month 12.  The custom curve is the dollar value of the prepayment by month
        # as a percentage of face value; we need the percentage of the loan balance that
        # that will be prepaid, so divide prepay_curve by outstanding balance, then 
        # renormalize so prepay[11] = prepay_risk

        eom_balance = 1 - np.cumsum(ppmt)
        adj_prepay_curve = prepay_curve / eom_balance
        adj_risk = adj_prepay_curve[:12].sum()
        prepay_curve *= prepay_risk / adj_risk

        # Note: the time=0 cash-flow is the initial borrowed amount, needed for the irr calc
        payments = np.zeros(term+1)
        payments_after_tax = np.zeros(term+1)
      
        cummulative_defaults = np.cumsum(default_curve)
        pct_prepaid_so_far = 0
        pct_defaulted_so_far = 0

        # add monthly payments
        # Note here that defaults are as a percentage of outstanding balance, 
        # whereas prepays are a percentage of original face value.
        for m in range(1, term+1):
            # calculate contractually-required payments
            contract_interest_due = ipmt[m-1] 
            contract_principal_due = ppmt[m-1] 
            eom_balance = ppmt[m:].sum()
            bom_balance = ppmt[m-1:].sum()

            # account for defaults, which happen before payments
            default_percent_this_month = default_curve[m-1] 
            pct_defaulted_so_far = cummulative_defaults[m-1]
            default_amount_this_month = bom_balance * default_percent_this_month

            # account for principal payments 
            performing_pct = (1 - pct_defaulted_so_far - pct_prepaid_so_far)
            interest_received = contract_interest_due * performing_pct 
            scheduled_principal_received = contract_principal_due * performing_pct 

            prepay_amount_this_month = prepay_curve[m-1] 
            if eom_balance > 0:
                prepay_percent_this_month = prepay_amount_this_month / eom_balance
            else:
                prepay_percent_this_month = 0.0
            pct_prepaid_so_far += prepay_percent_this_month

            payments[m] = interest_received + scheduled_principal_received + prepay_amount_this_month

            taxes_this_month = interest_received * net_payment_pct * self.income_tax_rate 
            taxes_this_month = taxes_this_month - default_amount_this_month * self.write_off_tax_shield
            payments_after_tax[m] = payments[m] - taxes_this_month 
            
        # reduce payments by lending club service charge
        payments *= net_payment_pct

        # Add initial investment outflow at time=0 to calculate irr: 
        payments[0] += -1
        payments_after_tax[0] += -1
        
        # the 12.0 is to convert back to annual rates.
        irr = 12.0 * np.irr(payments)
        irr_after_tax = 12.0 * np.irr(payments_after_tax)
        
        return irr, irr_after_tax
   

class DefaultCurve(object):
    ''' the i-th element in the baseline default curve is assumed to be
    the percentage of face value that has defaulted before the i-th scheduled 
    payment has been made'''
    grades = list('ABCDEFG')
    terms = [36,60]

    def __init__(self, model_dir=None):
        self.baseline_curves = dict()
        self._raw_curves = dict()
        self.model_dir = model_dir 
        if self.model_dir is not None:
            self.load_baseline_curves()

    def load_baseline_curves(self):
        curve_file = os.path.join(self.model_dir, 'default_curves.json')
        if os.path.exists(curve_file):
            self.baseline_curves = json.load(open(curve_file, 'r'))
        else:
            print('Baseline default curves not found')

    @staticmethod
    def get_label(grade, term):
        return '{}, {}'.format(grade.upper(), term)
 
    @staticmethod
    def convert_cummulative_to_marginal(curve):
        return np.diff(np.r_[[0], curve])

    @staticmethod
    def convert_marginal_to_cummulative(curve):
        return np.cumsum(curve)

    def get(self, grade, term, type='cummulative'):
        if type=='cummulative':
            return self.get_cummulative_curve(grade, term)
        elif type=='marginal':
            return self.get_marginal_curve(grade, term)
        elif type=='conditional':
            return self.get_conditional_curve(grade, term)
        elif type=='raw':
            return self.get_raw_curve(grade, term)

    def get_cummulative_curve(self, grade, term):
        curve = self.baseline_curves[self.get_label(grade, term)]
        return self.convert_marginal_to_cummulative(curve) 

    def get_marginal_curve(self, grade, term):
        curve = self.baseline_curves[self.get_label(grade, term)]
        return np.array(curve) 

    def get_conditional_curve(self, grade, term):
        ''' calculates the probability of default by month, as a percentage
        of the outstanding balance '''
        curve = self.get_marginal_curve(grade, term)
        pct_outstanding = np.linspace(1, 0, term+2)[:-1]
        return curve / pct_outstanding 
  
    def custom_curve(self, term, risk_12m):
        ''' default risk input is the estimated probability of default within
        12 months of issuance '''
        baseline_risks = np.array([self.get_cummulative_curve(grade, term)[11] 
                                   for grade in self.grades])
        idx = np.digitize(risk_12m, baseline_risks)

        if idx==0:
            factor = risk_12m / baseline_risks[idx]
            custom_curve = factor * self.get_marginal_curve(self.grades[idx], term)
        elif idx==len(baseline_risks):
            base = self.get_cummulative_curve(self.grades[idx-1], term)
            lower = self.get_cummulative_curve(self.grades[idx-2], term)
            incr = np.maximum(0, base - lower)
            factor = (risk_12m - baseline_risks[idx-1]) / incr[11]
            custom_cummulative_curve = base + factor * incr
            custom_curve = self.convert_cummulative_to_marginal(custom_cummulative_curve)
        else:
            lower = self.get_cummulative_curve(self.grades[idx-1], term)
            upper = self.get_cummulative_curve(self.grades[idx], term)
            incr = np.maximum(0, upper - lower)
            factor = (risk_12m - baseline_risks[idx-1]) / incr[11]
            custom_cummulative_curve = lower + factor * incr
            custom_curve = self.convert_cummulative_to_marginal(custom_cummulative_curve)
        return np.maximum(0, custom_curve)
  
    def plot_curves(self, curve_type='marginal'):
        if len(self.baseline_curves)>0:
            for term in self.terms: 
                plt.figure()
                legend = list()
                for grade in self.grades:
                    curve = self.get(grade, term, curve_type)
                    plt.plot(curve)
                    legend.append(self.get_label(grade, term))
                plt.title('{} Default Curves for Term={} by Grade'.format(curve_type.capitalize(), term))
                plt.legend(legend)
                plt.grid()
                plt.show()

    def estimate_from_payments(self):
        '''default curves'''
        print('Loading Payments file... this takes a while')
        cols = ['LOAN_ID', 'PBAL_BEG_PERIOD', 'PBAL_END_PERIOD', 'MONTHLYCONTRACTAMT', 'InterestRate', 
                'VINTAGE', 'IssuedDate', 'RECEIVED_AMT', 'DUE_AMT', 'PERIOD_END_LSTAT', 'MOB', 'term', 'grade']
        df = loanstats.load_payments(cols)
        print('Payments loaded')

        df['prepay_amt'] = np.maximum(0, df['RECEIVED_AMT'] - df['DUE_AMT'])
        df['delinquent_amt'] = np.maximum(0, -(df['RECEIVED_AMT'] - df['DUE_AMT']))
        df['mob_if_current'] = 0
        idx = (df.PERIOD_END_LSTAT=='Current') | (df.PERIOD_END_LSTAT == 'Fully Paid')
        df.ix[idx, 'mob_if_current'] = df.ix[idx, 'MOB']

        g_id = df.groupby('LOAN_ID')
        first = g_id.first()
        last = g_id.last()

        data = first[['PBAL_BEG_PERIOD', 'InterestRate', 'MONTHLYCONTRACTAMT', 
            'VINTAGE', 'IssuedDate', 'term', 'grade']].copy()
        data['last_status'] = last['PERIOD_END_LSTAT']
        data['last_balance'] = last['PBAL_END_PERIOD']
        data['age'] = last['MOB']

        data['last_current_mob'] = g_id['mob_if_current'].max()
        data['max_prepayment'] = g_id['prepay_amt'].max()
        data['max_delinquency'] = g_id['delinquent_amt'].max()

        default_status = ['Charged Off', 'Default', 'Late (31-120 days)']
        g = data.groupby(['IssuedDate', 'term', 'grade'])

        summary = dict()
        for k in g.groups.keys():
            v = g.get_group(k)
            max_age = min(k[1], v['age'].max())
            N = len(v)
            default_mob = v.ix[v.last_status.isin(default_status), 'last_current_mob'].values
            c = Counter(default_mob) 
            default_counts = sorted(c.items(), key=lambda x:x[0])
            summary[k] = (N, max_age, default_counts) 

        defaults = np.zeros((len(summary), 63), dtype=np.int)
        defaults[:,0] = [v[0] for v in summary.values()]
        defaults[:,1] = [v[1] for v in summary.values()]
        index = pd.MultiIndex.from_tuples(summary.keys(), names=['issue_month', 'term', 'grade'])

        issued = defaults.copy()
        for i, v in enumerate(summary.values()):
            issued[i,2:3+v[1]] = v[0]
            for months_paid, num in v[2]:
               defaults[i, 2+min(v[1], months_paid)] = num
            
        cols = ['num_loans', 'max_age'] + list(range(61))
        defaults = pd.DataFrame(data=defaults, index=index, columns=cols).reset_index()   
        issued = pd.DataFrame(data=issued, index=index, columns=cols).reset_index()    
        g_default = defaults.groupby(['term', 'grade']).sum()
        g_issued = issued.groupby(['term', 'grade']).sum()
        default_rates = (g_default / g_issued).ix[:, 0:]

        for lbl, row in default_rates.iterrows():
            term, grade = lbl 
            win = 19 if term==36 else 29 
            empirical_default = row[:term+1].values
            smoothed_default = scipy.signal.savgol_filter(empirical_default, win , 3)
            smoothed_default = np.maximum(0, smoothed_default) 
            curve_label = self.get_label(grade, term)
            self.baseline_curves[curve_label] = list(smoothed_default)
            self._raw_curves[curve_label] = list(empirical_default)

        default_curves_fname = os.path.join(paths.get_dir('training'), 'default_curves.json')
        json.dump(self.baseline_curves, open(default_curves_fname, 'w'), indent=4)
        print('Default curves estimated and saved')



class PrepayCurve(object):
    ''' the i-th element in the baseline prepay curve is assumed to be
    the percentage of face value that has prepaid before the i-th scheduled 
    payment has been made'''
    revol_util_buckets = np.array([10,20,30,40,50])
    bucket_labels = {0:'<10%', 1:'10%-20%', 2:'20%-30%',3:'30%-40%',4:'40%-50%', 5:'>50%'}
    terms = [36,60]

    def __init__(self, model_dir=None):
        self.baseline_curves = dict()
        self._raw_curves = dict()
        self.model_dir = model_dir 
        if self.model_dir is not None:
            self.load_baseline_curves()

    def load_baseline_curves(self):
        curve_file = os.path.join(self.model_dir, 'prepay_curves.json')
        if os.path.exists(curve_file):
            self.baseline_curves = json.load(open(curve_file, 'r'))
        else:
            print('Baseline prepay curves not found')

    @classmethod
    def get_revol_util_bucket(cls, revol_util):
        return int(np.digitize(revol_util, cls.revol_util_buckets))
       
    @classmethod
    def get_label(cls, revol_util, term):
        ''' revol_util can be a numerical value, or the bucket label string 
        defined in the class variables'''
        if revol_util in cls.bucket_labels.values():
            bucket_label = revol_util
        else:
            bucket_index = cls.get_revol_util_bucket(revol_util) 
            bucket_label = cls.bucket_labels[bucket_index]
        return '{}, {}'.format(bucket_label, term)
 
    def get_curve_by_key(self, key, term):
        curve_label = self.get_label(self.bucket_labels[key], term)
        return self.baseline_curves[curve_label]

    @staticmethod
    def convert_cummulative_to_marginal(curve):
        return np.diff(np.r_[[0], curve])

    @staticmethod
    def convert_marginal_to_cummulative(curve):
        return np.cumsum(curve)

    def get(self, revol_util, term, type='cummulative'):
        if type=='cummulative':
            return self.get_cummulative_curve(revol_util, term)
        elif type=='marginal':
            return self.get_marginal_curve(revol_util, term)
        elif type=='conditional':
            return self.get_conditional_curve(revol_util, term)
        elif type=='raw':
            return self.get_raw_curve(revol_util, term)

    def get_marginal_curve(self, revol_util, term):
        curve = self.baseline_curves[self.get_label(revol_util, term)]
        return np.array(curve)

    def get_raw_curve(self, revol_util, term):
        curve_label = self.get_label(revol_util, term)
        if curve_label in self._raw_curves.keys():
            curve = self._raw_curves[self.get_label(revol_util, term)]
        else:
            curve = []
        return np.array(curve)

    def get_cummulative_curve(self, revol_util, term):
        curve = self.get_marginal_curve(revol_util, term)
        return self.convert_marginal_to_cummulative(curve) 

    def get_conditional_curve(self, revol_util, term):
        ''' calculates the probability of prepayment by month, as a percentage
        of the outstanding balance; assumes pricipal is repaid linearly '''
        curve = self.get_marginal_curve(revol_util, term)
        pct_outstanding = np.linspace(1, 0, term+1)[:-1]
        return curve / pct_outstanding 
 
    def get_curve_range(self, term):
        all_curves = np.vstack((self.get_curve_by_key(key, term) for key in self.bucket_labels.keys()))
        return all_curves.max(0) - all_curves.min(0)

    def custom_curve(self, term, risk_12m):
        ''' risk input is the estimated percent of face value that will be prepaid 
        within 12 months of issuance '''
        baseline_risks = [self.get_cummulative_curve(self.bucket_labels[lbl], term)[11] 
                          for lbl in sorted(self.bucket_labels.keys())]
        idx = int(np.digitize(risk_12m, baseline_risks))

        if idx==0:
            base = self.get_marginal_curve(self.bucket_labels[idx], term)
            incr = self.get_curve_range(term) 
            base_risk = self.convert_marginal_to_cummulative(base)[11]
            incr_risk = self.convert_marginal_to_cummulative(incr)[11] 
            factor = (risk_12m - base_risk) / incr_risk
            custom_curve = base + factor * incr 
        elif idx==len(baseline_risks):
            base = self.get_marginal_curve(self.bucket_labels[idx-1], term)
            factor = (risk_12m / baseline_risks[idx-1]) 
            custom_curve = factor * base 
        else:
            upper = self.get_cummulative_curve(self.bucket_labels[idx-1], term)
            lower = self.get_cummulative_curve(self.bucket_labels[idx], term)
            incr = upper - lower
            factor = (risk_12m - baseline_risks[idx]) / incr[11]
            custom_cummulative_curve = lower + factor * incr
            custom_curve = self.convert_cummulative_to_marginal(custom_cummulative_curve)

        return np.maximum(0, custom_curve)
            
  
    def plot_curves(self, curve_type='marginal'):
        if len(self.baseline_curves)>0:
            for term in self.terms: 
                plt.figure()
                legend = list()
                for revol_util in self.bucket_labels.values():
                    curve = self.get(revol_util, term, curve_type)
                    plt.plot(curve)
                    legend.append(self.get_label(revol_util, term))
                plt.title('{} Prepayment Curves for Term={} by RevolUtil'.format(curve_type.capitalize(), term))
                plt.legend(legend, loc=0)
                plt.grid()
                plt.show()


    def estimate_from_payments(self):
        print('Loading Payments file... this takes a while')
        cols = ['LOAN_ID', 'PBAL_BEG_PERIOD', 'MONTHLYCONTRACTAMT', 'InterestRate', 
                'VINTAGE', 'IssuedDate', 'RECEIVED_AMT', 'DUE_AMT', 'PERIOD_END_LSTAT', 'MOB', 'term', 'grade', 
                'dti', 'HomeOwnership', 'MonthlyIncome', 'EmploymentLength', 'RevolvingLineUtilization']
        df = loanstats.load_payments(cols)
        print('Payments loaded')

        g_id = df.groupby('LOAN_ID')
        df['prepay_amt'] = np.maximum(0, df['RECEIVED_AMT'] - df['DUE_AMT'])

        df['loan_amount'] = g_id['PBAL_BEG_PERIOD'].transform(lambda x: x.iloc[0])
        df['prepay_pct'] = df['prepay_amt'] / df['loan_amount'] 
        df['issue_year'] = df['IssuedDate'].apply(lambda x: int(x[-4:]))
        prepays = df.pivot(index='LOAN_ID', columns='MOB', values='prepay_pct') 

        # combine all payments for MOB=0 (a very small number of prepayments 
        # happen before the first payment is due) with MOB=1
        if 0 in prepays.columns:
            prepays[1] = prepays[1] + prepays[0].fillna(0)
            prepays = prepays.ix[:,1:]

        join_cols = ['term', 'grade', 'IssuedDate', 'MOB']
        prepays = prepays.join(g_id[join_cols].last())

        # revUtil of, e.g., 45% is 0.45 in PMTs file, but 45 in API; need to convert 
        revol_util = 100 * g_id['RevolvingLineUtilization'].first().fillna(0) 
        revol_util.name = 'revol_util'
        prepays = prepays.join(revol_util)

        # clean the prepayment rows for each issue date
        prepays = prepays.sort_values(by=['IssuedDate'])
        for d in set(prepays['IssuedDate'].values):
            idx = prepays.IssuedDate == d
            max_mob = prepays.ix[idx, 'MOB'].max()
            prepays.ix[idx, (max_mob-1)] = np.NaN  #ignore data from most recent month, as payments are incomplete
            prepays.ix[idx, :(max_mob-1)] = prepays.ix[idx, :(max_mob-1)].fillna(0)

        prepays['revol_util_grp'] = prepays['revol_util'].apply(self.get_revol_util_bucket)
        g_prepays = prepays.groupby(['revol_util_grp', 'term'])
        mean_prepays = g_prepays.mean()

        # smooth the curves
        begin_smooth = 0
        for idx, row in mean_prepays.iterrows():
            revol_util_grp, term = idx  
            win = 7 if term==36 else 13 
            empirical_prepays = row[:term-1].values
            smoothed_prepays = scipy.signal.savgol_filter(empirical_prepays[begin_smooth:], win , 3)
            smoothed_prepays = np.maximum(0, smoothed_prepays) 
            smoothed_prepays = np.r_[empirical_prepays[:begin_smooth], smoothed_prepays]
            smoothed_prepays = np.r_[smoothed_prepays, [0]]            #force prepayment on due date to be zero
            curve_label = self.get_label(self.bucket_labels[revol_util_grp], term)
            self.baseline_curves[curve_label] = list(smoothed_prepays)
            self._raw_curves[curve_label] = empirical_prepays

        # key values need to be strings to dump to json file
        prepay_curves_fname = os.path.join(paths.get_dir('training'), 'prepay_curves.json')
        json.dump(self.baseline_curves, open(prepay_curves_fname, 'w'), indent=4)
        print('Prepay curves estimated and saved')
       
        

if __name__ == '__main__':
    DefaultCurve().estimate_from_payments()
    PrepayCurve().estimate_from_payments()
