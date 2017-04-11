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
            print 'Baseline default curves not found'

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
        print 'Loading Payments file... this takes a while'
        cols = ['LOAN_ID', 'PBAL_BEG_PERIOD', 'PBAL_END_PERIOD', 'MONTHLYCONTRACTAMT', 'InterestRate', 
                'VINTAGE', 'IssuedDate', 'RECEIVED_AMT', 'DUE_AMT', 'PERIOD_END_LSTAT', 'MOB', 'term', 'grade']
        df = loanstats.load_payments(cols)
        print 'Payments loaded'

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
            
        cols = ['num_loans', 'max_age'] + range(61)
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
            print 'Baseline prepay curves not found'

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
 
    def custom_curve(self, term, risk_12m):
        ''' risk input is the estimated percent of face value that will be prepaid 
        within 12 months of issuance '''
        
        baseline_risks = [self.get_cummulative_curve(self.bucket_labels[lbl], term)[11] 
                          for lbl in sorted(self.bucket_labels.keys())]
        idx = int(np.digitize(risk_12m, baseline_risks))

        if idx==0:
            factor = risk_12m / baseline_risks[idx]
            custom_curve = factor * self.get_marginal_curve(self.bucket_labels[idx], term)
        elif idx==len(baseline_risks):
            higher = self.get_cummulative_curve(self.bucket_labels[idx-1], term)
            base = self.get_cummulative_curve(self.bucket_labels[idx-2], term)
            incr = higher - base
            factor = (risk_12m - baseline_risks[idx-2]) / incr[11]
            custom_cummulative_curve = base + factor * incr
            custom_curve = self.convert_cummulative_to_marginal(custom_cummulative_curve)
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
        print 'Loading Payments file... this takes a while'
        cols = ['LOAN_ID', 'PBAL_BEG_PERIOD', 'MONTHLYCONTRACTAMT', 'InterestRate', 
                'VINTAGE', 'IssuedDate', 'RECEIVED_AMT', 'DUE_AMT', 'PERIOD_END_LSTAT', 'MOB', 'term', 'grade', 
                'dti', 'HomeOwnership', 'MonthlyIncome', 'EmploymentLength', 'RevolvingLineUtilization']
        df = loanstats.load_payments(cols)
        print 'Payments loaded'

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
        prepays = prepays.sort('IssuedDate')
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
       
        

#TODO: modify this to use the rev_util version of prepayment curves, and adjust the curves
# by hinge tilting them from month 20 & 30 for 3-year and 5-year loans, resp.
# Hinge the default curves from month 15 & 20 for 3- and 5-year, resp.
# Those hinge points roughly match empirical observations for how curves move from
# one grade to the nearby ones.
class ReturnCalculator(object):
    def __init__(self, default_curves, prepay_curves):
        self.default_curves = default_curves
        self.prepay_curves = prepay_curves  #This is not currently used

    def calc_irr(self, loan, default_rate_12m, prepayment_rate_12m):
        ''' All calculations assume a loan amount of $1.
        Note the default curves are the cumulative percent of loans that have defaulted prior 
        to month m; the prepayment rate is the percentage of loans that were prepaid prior to 12m.
        We'll assume that the prepayments are in full.  In the code below, think of the calculations
        as applying to a pool of 100 loans, with a percentage fully prepaying each month and
        a percentage defaulting each month.'''

        net_payment_pct = 0.99  #LC charges 1% fee on all incoming payments
        income_tax_rate = 0.5
        capital_gains_tax_rate = 0.2

        key = '{}{}'.format(min('G', loan['grade']), loan['term']) 
        base_cdefaults = np.array(self.default_curves[key])
        risk_factor = default_rate_12m / base_cdefaults[11]
        loan['risk_factor'] = risk_factor

        # adjust only the first 15 months of default rates downward to match the model's 12-month default estimate.
        # this is a hack; adjusting the entire curve seems obviously wrong.  E.g if we had a C default curve
        # that was graded D, adjusting the entire D curve down based on the 12-mth ratio would underestimate defaults
        cdefaults = np.r_[base_cdefaults[:1],np.diff(base_cdefaults)]
        if risk_factor < 1.0:
            cdefaults[:15] *= risk_factor
        else:
            cdefaults *= risk_factor

        cdefaults = cdefaults.cumsum()
        eventual_default_pct = cdefaults[-1]

        max_prepayment_pct = 1 - eventual_default_pct

        # catch the case where total prepayments + total defaults > 100%  (they're estimated independently)
        if max_prepayment_pct <= prepayment_rate_12m: 
            return 0, 0
  
        #TODO: this isn't using the prepayment curves; FIX
        # prepayment model give the odds of full prepayment in the first 12 months 
        # here we calculate the probability of prepayment just for the loans that 
        # won't default
        prepayment_pool_decay_12m = (max_prepayment_pct - prepayment_rate_12m) / max_prepayment_pct
        prepay_rate = 1.0 - prepayment_pool_decay_12m ** (1/12.0)  

        monthly_int_rate = loan['intRate']/1200.
        contract_monthly_payment = loan['monthly_payment'] / loan['loan_amount']
        current_monthly_payment = contract_monthly_payment

        # start with placeholder for time=0 investment for irr calc later
        payments = np.zeros(loan['term']+1)
        payments_after_tax = np.zeros(loan['term']+1)
        
        contract_principal_balance = 1.0
        pct_loans_prepaid = 0.0
        pct_loans_defaulted = 0.0
        # add monthly payments
        for m in range(1, loan['term']+1):
            
            # calculate contractually-required payments
            contract_interest_due = contract_principal_balance * monthly_int_rate
            contract_principal_due = min(contract_principal_balance, 
                                         contract_monthly_payment - contract_interest_due)

            default_rate_this_month = cdefaults[m-1] - pct_loans_defaulted
            pct_loans_defaulted = cdefaults[m-1]
     
            # account for defaults and prepayments 
            performing_pct = (1 - pct_loans_defaulted - pct_loans_prepaid)
            interest_received = contract_interest_due * performing_pct 
            scheduled_principal_received = contract_principal_due * performing_pct 
            scheduled_principal_defaulted = default_rate_this_month * contract_principal_balance

            # update contractual principal remaining (i.e. assuming no prepayments or defaults) 
            # prior to calculating prepayments
            contract_principal_balance -= contract_principal_due

            #prepayments are a fixed percentage of the remaining pool of non-defaulting loans
            prepayment_pct = max(0, (max_prepayment_pct - pct_loans_prepaid)) * prepay_rate
            prepayment_amount = contract_principal_balance * prepayment_pct 

            # account for prepayments
            pct_loans_prepaid += prepayment_pct 

            payments[m] = interest_received + scheduled_principal_received + prepayment_amount

            taxes = interest_received * net_payment_pct * income_tax_rate 
            taxes = taxes - scheduled_principal_defaulted * capital_gains_tax_rate
            payments_after_tax[m] = payments[m] - taxes 
            
        # reduce payments by lending club service charge
        payments *= net_payment_pct

        # Add initial investment outflow at time=0 to calculate irr: 
        payments[0] += -1
        payments_after_tax[0] += -1
        irr = np.irr(payments)
        irr_after_tax = np.irr(payments_after_tax)
        
        # use same units for irr as loan interest rate
        annualized_irr = irr * 12.0
        annualized_irr_after_tax = irr_after_tax * 12.0

        return annualized_irr, annualized_irr_after_tax 
    

