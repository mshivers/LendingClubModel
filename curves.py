## Lending Club's prepayment curves are all percentages of face value by month, 
## rather than percentage of outstanding face value, which we do.  Should we change?
import os
import json
from matplotlib import pyplot as plt

class DefaultCurve(object):
    grades = list('ABCDEFG')
    terms = [36,60]

    def __init__(self, model_dir=None):
        self.baseline_curves = dict()
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
    def curve_label(grade, term):
        return '{}{}'.format(grade, term)

    def get_default_curve(self, grade, term, default_risk):
        ''' default risk input is the estimated probability of default within
        12 months of issuance '''
        pass

    def plot_curves(self):
        if len(self.baseline_curves)>0:
            for term in self.terms: 
                plt.figure()
                legend = list()
                for grade in self.grades:
                    label = self.curve_label(grade,term)
                    curve = self.baseline_curves[label]
                    plt.plot(curve)
                    legend.append(label)
                plt.title('Default Curves for Term={}'.format(term))
                plt.legend(legend)
                plt.grid()
                plt.show()

    def estimate_baseline_curves():
        now = dt.now
        pd.set_option('display.max_colwidth', 200)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', 100)
        t = now()

        cols = ['LOAN_ID', 'PBAL_BEG_PERIOD', 'PBAL_END_PERIOD', 'MONTHLYCONTRACTAMT', 'InterestRate', 
                'VINTAGE', 'IssuedDate', 'RECEIVED_AMT', 'DUE_AMT', 'PERIOD_END_LSTAT', 'MOB', 'term', 'grade']

        df = pd.read_csv(payments_file, sep=',', usecols=cols)

        print 'After read_csv',  (now() - t).total_seconds()
        df['prepay_amt'] = np.maximum(0, df['RECEIVED_AMT'] - df['DUE_AMT'])
        df['delinquent_amt'] = np.maximum(0, -(df['RECEIVED_AMT'] - df['DUE_AMT']))
        df['mob_if_current'] = 0
        idx = (df.PERIOD_END_LSTAT=='Current') | (df.PERIOD_END_LSTAT == 'Fully Paid')
        df.ix[idx, 'mob_if_current'] = df.ix[idx, 'MOB']

        g_id = df.groupby('LOAN_ID')
        print 'After groupby by ID', (now() - t).total_seconds()
        first = g_id.first()
        last = g_id.last()

        print 'After first/last', (now() - t).total_seconds()
        data = first[['PBAL_BEG_PERIOD', 'InterestRate', 'MONTHLYCONTRACTAMT', 
            'VINTAGE', 'IssuedDate', 'term', 'grade']].copy()
        data['last_status'] = last['PERIOD_END_LSTAT']
        data['last_balance'] = last['PBAL_END_PERIOD']
        data['age'] = last['MOB']

        data['last_current_mob'] = g_id['mob_if_current'].max()
        data['max_prepayment'] = g_id['prepay_amt'].max()
        data['max_delinquency'] = g_id['delinquent_amt'].max()

        data = data.rename(columns=lambda x: x.lower())

        default_status = ['Charged Off', 'Default', 'Late (31-120 days)']
        g = data.groupby(['issueddate', 'term', 'grade'])

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

        defaults['grade'] = np.minimum(defaults['grade'], 'G')
        issued['grade'] = np.minimum(issued['grade'], 'G')

        g_default = defaults.groupby(['term', 'grade']).sum()
        g_issued = issued.groupby(['term', 'grade']).sum()
        default_rates = (g_default / g_issued).ix[:, 0:]

        default_curves = dict()
        for i, r in default_rates.iterrows():
            N = i[0]
            win = 19 if N==36 else 29 
            empirical_default = r[:N+1].values
            smoothed_default = scipy.signal.savgol_filter(empirical_default, win , 3)
            smoothed_default = np.maximum(0, smoothed_default) 
            default_curves['{}{}'.format(i[1], i[0])] = list(np.cumsum(smoothed_default))

        for term in [36,60]:
            plt.figure()
            for grade in all_grades:
                data = np.diff(np.r_[[0], default_curves['{}{}'.format(grade,term)]])
                plt.plot(data)
            plt.title('Default Curves for Term={}'.format(term))
            plt.grid()
            plt.show()


        for term in [36,60]:
            plt.figure()
            for grade in all_grades:
                plt.plot(default_curves['{}{}'.format(grade,term)])
            plt.title('Term: {}'.format(term))
            plt.grid()
            plt.show()
        
        training_data_dir = paths.get_dir('training')
        default_curves_fname = os.path.join(training_data_dir, 'default_curves.json')
        json.dump(default_curves, open(default_curves_fname, 'w'), indent=4)


def estimate_prepay_curves():
    now = dt.now
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 100)
    t = now()

    cols = ['LOAN_ID', 'PBAL_BEG_PERIOD', 'PBAL_END_PERIOD', 'MONTHLYCONTRACTAMT', 'InterestRate', 
            'VINTAGE', 'IssuedDate', 'RECEIVED_AMT', 'DUE_AMT', 'PERIOD_END_LSTAT', 'MOB', 'term', 'grade', 
            'dti', 'HomeOwnership', 'MonthlyIncome', 'EmploymentLength', 'RevolvingLineUtilization']
    df = pd.read_csv(payments_file, sep=',', usecols=cols)

    print 'After read_csv',  (now() - t).total_seconds()
    df['prepay_amt'] = np.maximum(0, df['RECEIVED_AMT'] - df['DUE_AMT'])

    g_id = df.groupby('LOAN_ID')
    print 'After groupby by ID', (now() - t).total_seconds()

    last = g_id.last()
    first = g_id.first()
    print 'After last()', (now() - t).total_seconds()
        
    df['bal_before_prepayment'] = df['PBAL_END_PERIOD'] + np.maximum(0, df['RECEIVED_AMT'] - df['DUE_AMT'])

    # denominator is to deal with the corner case when the last payment month has a stub payment
    df['prepay_pct'] = df['prepay_amt'] / np.maximum(df['DUE_AMT'], df['bal_before_prepayment'])
    df['issue_year'] = df['IssuedDate'].apply(lambda x: int(x[-4:]))
    prepays = df.pivot(index='LOAN_ID', columns='MOB', values='prepay_pct') 

    # combine all payments for MOB=0 (a very small number of prepayments before the first payment is due) with MOB=1
    prepays[1] = prepays[1] + prepays[0].fillna(0)
    del prepays[0]

    join_cols = ['term', 'grade', 'IssuedDate', 'MOB']
    prepays = prepays.join(last[join_cols])

    revol_util = first['RevolvingLineUtilization'].fillna(0)
    revol_util.name = 'revol_util'
    prepays = prepays.join(revol_util)

    prepays = prepays.sort('IssuedDate')
    for d in sorted(list(set(prepays['IssuedDate'].values))):
        idx = prepays.IssuedDate == d
        max_mob = prepays.ix[idx, 'MOB'].max()
        prepays.ix[idx, :(max_mob-1)] = prepays.ix[idx, :(max_mob-1)].fillna(0)
        print d, max_mob

    prepays['revol_util_grp'] = prepays['revol_util'].apply(lambda x: min(5, int(10*x)))
    g_prepays = prepays.groupby(['term', 'revol_util_grp'])

    mean_prepays = g_prepays.mean()

    prepay_curves = dict()
    begin_smooth = 0
    for N in [36, 60]:
        for i, r in mean_prepays.iterrows():
            if N == i[0]:
                win = 7 if N==36 else 13 
                empirical_prepays = r[:N].values
                smoothed_prepays = scipy.signal.savgol_filter(empirical_prepays[begin_smooth:], win , 3)
                smoothed_prepays = np.maximum(0, smoothed_prepays) 
                smoothed_prepays = np.r_[empirical_prepays[:begin_smooth], smoothed_prepays]
                prepay_curves[i] = list(smoothed_prepays)

    for term in [36,60]:
        plt.figure()
        legend = list()
        for k in sorted(prepay_curves.keys()):
            if k[0] == term:
                legend.append(k)
                plt.plot(prepay_curves[k])
        plt.title(term)
        plt.legend(legend, loc=1)
        plt.grid()
        plt.show()

    # key values need to be strings to dump to json file
    training_data_dir = paths.get_dir('training')
    prepay_curves = dict((str(k),v) for k,v in prepay_curves.items())
    prepay_curves_fname = os.path.join(training_data_dir, 'prepay_curves_rev_util.json')
    json.dump(prepay_curves, open(prepay_curves_fname, 'w'), indent=4)
   


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
    

