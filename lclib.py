import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from collections import defaultdict, Counter
import scipy.signal
from matplotlib import pyplot as plt
import utils
from personalized import p
from data_classes import ReferenceData, StringToNumberConverter, APIDataParser, PathManager

LARGE_INT = 9999 
NEGATIVE_INT = -1 
update_hrs = [9,13,17,21]  #Times new loans are posted in NY time


def load_training_data(regen=False):
    if os.path.exists(cached_training_data_file) and not regen:
        update_dt = dt.fromtimestamp(os.path.getmtime(cached_training_data_file))
        days_old = (dt.now() - update_dt).days 
        print 'Using cached LC data created on {}; cache is {} days old'.format(update_dt, days_old)
        df = pd.read_csv(cached_training_data_file)
    else:
        print 'Cache not found. Generating cache from source data'
        cache_training_data()
        df = load_training_data()
    return df

def cache_training_data():
    #rename columns to match API fields
    ref_data = ReferenceData() 
    col_name_map = ref_data.get_loanstats2api_map()
    training_data_dir = PathManager.get_dir('training')

    def clean_raw_data(df):
        df = df.rename(columns=col_name_map)
        idx1 = ~(df[['last_pymnt_d', 'issue_d', 'annualInc']].isnull()).any(1)
        df = df.ix[idx1].copy()
        df['issue_d'] = df['issue_d'].apply(lambda x: dt.strptime(x, '%b-%Y'))
        idx2 = df['issue_d']>=np.datetime64('2013-10-01')
        idx3 = df['issue_d'] <= dt.now() - td(days=366)
        df = df.ix[idx2&idx3].copy()
        df['id'] = df['id'].astype(int)
        return df

    fname = os.path.join(loanstats_dir, 'LoanStats3{}_securev1.csv')
    fname2 = os.path.join(loanstats_dir, 'LoanStats_securev1_{}.csv')
    dataframes = list()
    print 'Importing Raw Data Files'
    dataframes.append(clean_raw_data(pd.read_csv(fname.format('a'), header=1)))
    dataframes.append(clean_raw_data(pd.read_csv(fname.format('b'), header=1)))
    dataframes.append(clean_raw_data(pd.read_csv(fname.format('c'), header=1)))
    dataframes.append(clean_raw_data(pd.read_csv(fname.format('d'), header=1)))
    dataframes.append(clean_raw_data(pd.read_csv(fname2.format('2016Q1'), header=1)))
    #dataframes.append(clean_raw_data(pd.read_csv(fname2.format('2016Q2'), header=1)))
    #dataframes.append(clean_raw_data(pd.read_csv(fname2.format('2016Q3'), header=1)))
    print 'Concatenating dataframes'
    df = pd.concat(dataframes, ignore_index=True)
    print 'Dataframes imported'
 
    # clean dataframe
    cvt = dict()
    # intRate and revolUtil are the only fields where the format is "xx.x%" (with a % sign in the string)
    cvt['intRate'] = lambda x: float(x[:-1])
    cvt['revolUtil'] = lambda x: np.nan if '%' not in str(x) else round(float(x[:-1]),0)
    cvt['desc'] = lambda x: float(len(str(x)) > 3)
    cvt['last_pymnt_d'] = lambda x: dt.strptime(x, '%b-%Y')
    cvt['earliestCrLine'] = lambda x: dt.strptime(x, '%b-%Y')
    cvt['term'] = lambda x: int(x.strip().split(' ')[0])

    for col in cvt.keys():
        print 'Parsing {}'.format(col)
        df[col] = df[col].apply(cvt[col])

    api_parser = APIDataParser()
    for field in api_parser.null_fill_fields():
        if field in df.columns:
            fill_value = api_parser.null_fill_value(field)
            print 'Filling {} nulls with {}'.format(field, fill_value)
            df[field] = df[field].fillna(fill_value)

    string_converter = StringToNumberConverter()
    for col in string_converter.accepted_fields:
        if col in df.columns:
            print 'Converting {} string to numeric'.format(col)
            func = np.vectorize(lambda x: string_converter.convert(col, x))
            df[col] = df[col].apply(func)
   
    df['clean_title'] = df['empTitle'].apply(utils.clean_title)
    df['empTitle_length'] = df['empTitle'].apply(lambda x: len(x))

    # Calculate target values for various prediction models
    # add default info
    print 'Calculating Target model target values'
    df['wgt_default'] = 0.0 
    df.ix[df['loan_status']=='In Grace Period', 'wgt_default'] = 0.26
    df.ix[df['loan_status']=='Late (16-30 days)', 'wgt_default'] = 0.59
    df.ix[df['loan_status']=='Late (31-120 days)', 'wgt_default'] = 0.77
    df.ix[df['loan_status']=='Default', 'wgt_default'] = 0.86
    df.ix[df['loan_status']=='Charged Off', 'wgt_default'] = 1.0

    # we want to find payments strictly less than 1 year, so we use 360 days here.
    just_under_one_year = 360*24*60*60*1e9  
    time_to_last_pymnt = df['last_pymnt_d']-df['issue_d']
    df['12m_late'] = (df['wgt_default']>0) & (time_to_last_pymnt<just_under_one_year)
    df['12m_wgt_default'] = df['12m_late'] * df['wgt_default']

    # add prepayment info
    df['12m_prepay'] = 0.0
    # for Fully Paid, assume all prepayment happened in last month
    just_over_12months = 12.5*30*24*60*60*1e9  
    prepay_12m_idx = ((df['loan_status']=='Fully Paid') & (time_to_last_pymnt<=just_over_12months))
    df.ix[prepay_12m_idx, '12m_prepay'] = 1.0

    # partial prepays
    df['mob'] = np.ceil(time_to_last_pymnt.astype(int) / (just_over_12months / 12.0))
    prepayments = np.maximum(0, df['total_pymnt'] - df['mob'] * df['installment'])
    partial_12m_prepay_idx = (df['loan_status']=='Current') & (prepayments > 0)
    prepay_12m_pct = prepayments / df['loanAmount'] * (12. / np.maximum(12., df.mob))
    df.ix[partial_12m_prepay_idx, '12m_prepay'] = prepay_12m_pct[partial_12m_prepay_idx]

    # tag data for in-sample and oos (in sample issued at least 14 months ago. Issued 12-13 months ago is oos
    df['in_sample'] = df['issue_d'] < dt.now() - td(days=14*31)

    # process job title features
    sample = (df.in_sample)
    odds = OddsModel(tok_type='alltoks', string_name='empTitle', value_name='prepay')
    odds.fit(df.ix[sample, 'empTitle'].values, df.ix[sample, '12m_prepay'].values)
    odds.save(training_data_dir)
    feature_name = '{}_{}_odds'.format(odds.string_name, odds.value_name)
    df[feature_name] = df[odds.string_name].apply(odds.run)
    
    sample = (df.grade>=2) & (df.in_sample)
    odds = OddsModel(tok_type='alltoks', string_name='empTitle', value_name='default')
    odds.fit(df.ix[sample, 'empTitle'].values, df.ix[sample, '12m_wgt_default'].values)
    odds.save(training_data_dir)
    feature_name = '{}_{}_odds'.format(odds.string_name, odds.value_name)
    df[feature_name] = df[odds.string_name].apply(odds.run)

    #process frequency features
    freq = FrequencyModel(string_name='clean_title')
    freq.fit(df.ix[df.in_sample, 'clean_title'].values)
    freq.save(training_data_dir)
    df['{}_freq'.format(freq.string_name)] = df[freq.string_name].apply(freq.run)

    freq = FrequencyModel(string_name='empTitle')
    freq.fit(df.ix[df.in_sample, 'empTitle'].values)
    freq.save(training_data_dir)
    df['{}_freq'.format(freq.string_name)] = df[freq.string_name].apply(freq.run)

    ### Add non-LC features
    print 'Adding BLS data'
    urate = pd.read_csv(os.path.join(bls_data_dir, 'urate_by_3zip.csv'), index_col=0)
    ur = pd.DataFrame(np.zeros((len(urate),999))*np.nan,index=urate.index, columns=[str(i) for i in range(1,1000)])
    ur.ix[:,:] = urate.median(1).values[:,None]
    ur.ix[:,urate.columns] = urate
    avg_ur = pd.rolling_mean(ur, 12)
    ur_chg = ur - ur.shift(12)

    df['urate_d'] = df['issue_d'].apply(lambda x: int(str((x-td(days=60)))[:7].replace('-','')))
    df['urate'] = [ur[a][b] for a,b in zip(df['addrZip'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['avg_urate'] = [avg_ur[a][b] for a,b in zip(df['addrZip'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['urate_chg'] = [ur_chg[a][b] for a,b in zip(df['addrZip'].apply(lambda x: str(int(x))), df['urate_d'])]

    print 'Adding FHFA data'
    hpa4 = pd.read_csv(os.path.join(fhfa_data_dir, 'hpa4.csv'), index_col = 0)
    mean_hpa4 = hpa4.mean(1)
    missing_cols = [str(col) for col in range(0,1000) if str(col) not in hpa4.columns]
    for c in missing_cols:
        hpa4[c] = mean_hpa4

    df['hpa_date'] = df['issue_d'].apply(lambda x:x-td(days=120))
    df['hpa_qtr'] = df['hpa_date'].apply(lambda x: 100*x.year + x.month/4 + 1)
    df['hpa4'] = [hpa4.ix[a,b] for a,b in zip(df['hpa_qtr'], df['addrZip'].apply(lambda x: str(int(x))))]
     
    print 'Adding Census data'
    df['census_median_income'] = df['addrZip'].apply(ref_data.get_median_income)

    # Add calculated features 
    feature_calculator = FeatureCalculator()
    feature_calculator.compute(df)
    '''
    print 'Adding Computed features'
    one_year = 365*24*60*60*1e9
    df['credit_length'] = ((df['issue_d'] - df['earliestCrLine']).astype(int)/one_year)
    df['credit_length'] = df['credit_length'].apply(lambda x: max(-1,round(x,0)))
    df['even_loan_amnt'] = df['loanAmount'].apply(lambda x: float(x==round(x,-3)))
    df['int_pymt'] = df['loanAmount'] * df['intRate'] / 1200.0
    df['revol_bal-loan'] = df['revolBal'] - df['loanAmount']
    df['pct_med_inc'] = df['annualInc'] / df['census_median_income']
    df['pymt_pct_inc'] = df['installment'] / df['annualInc'] 
    df['revol_bal_pct_inc'] = df['revolBal'] / df['annualInc']
    df['int_pct_inc'] = df['int_pymt'] / df['annualInc'] 
    df['loan_pct_income'] = df['loanAmount'] / df['annualInc']
    df['cur_bal-loan_amnt'] = df['totCurBal'] - df['loanAmount'] 
    df['cur_bal_pct_loan_amnt'] = df['totCurBal'] / df['loanAmount'] 
    df['mort_bal'] = df['totCurBal'] - df['totalBalExMort']
    df['mort_pct_credit_limit'] = df['mort_bal'] * 1.0 / df['totHiCredLim']
    df['mort_pct_cur_bal'] = df['mort_bal'] * 1.0 / df['totCurBal']
    df['revol_bal_pct_cur_bal'] = df['revolBal'] * 1.0 / df['totCurBal']
    '''
    df.to_csv(PathManager.get_filepath('training_cache'))


class FrequencyModel(object):
    ''' Model that converts a string feature into the number of 
    instances of that string that exists in the historical data'''

    def __init__(self, freq_dict=None, string_name=None):
        if isinstance(freq_dict, dict):
            freq_dict = defaultdict(lambda :0, freq_dict)
        self.freq_dict = freq_dict
        self.string_name = string_name

    def fit(self, strings):
        '''Creates the dictionary with the frequency mapping'''
        counts = Counter(strings)
        self.freq_dict = defaultdict(lambda :0, counts.items())

    def run(self, input_string):
        return self.freq_dict[input_string]

    def is_fit(self):
        return isinstance(self.freq_dict, dict)

    def save(self, fname):
        config = {'freq_dict': dict(self.freq_dict), 
                  'string_name':self.string_name}
        if os.path.isdir(fname):
            file_name = '{}_frequency.json'.format(self.string_name)
            fname = os.path.join(fname, file_name) 
        json.dump(config, open(fname, 'w'))



class OddsModel(object):
    def __init__(self, tok_type, odds_dict=None, string_name=None, value_name=None):
        self.tok_type = tok_type
        if isinstance(odds_dict, dict):
            odds_dict = defaultdict(lambda :0, odds_dict)
        self.odds_dict = odds_dict 
        self.string_name = string_name #name of the string field to apply the model to.
        self.value_name = value_name #name of the value field the model predicts 
    
    def _get_all_substrings(self, input_string):
        length = len(input_string)
        return [input_string[i:j+1] for i in xrange(length) for j in xrange(i,length)]
       
    def _get_substrings_of_length(self, input_string, length):
        return [input_string[i:i+length+1] for i in range(max(1,len(input_string)-length))]

    def _get_words(self, input_string):
        return input_string.strip().split()

    def get_tokens(self, input_string):
        if self.tok_type=='word':
            toks = self._get_words(input_string) 
        elif self.tok_type=='phrase':
            toks = [input_string]
        elif self.tok_type=='alltoks':
            toks = self._get_all_substrings(input_string) 
        elif isinstance(self.tok_type, int):
            toks = self._get_substrings_of_length(input_string, self.tok_type) 
        else:
            raise Exception('unknown tok_type')
        return toks

    def fit(self, strings, values):
        ''' creates odds dictionary.  The tok_type field is either the length of the string token
        to use, or if tok_type='word', the tokens are entire words'''
        value_sum = defaultdict(lambda :0)
        tok_count = defaultdict(lambda :0)
        data = pd.DataFrame({'strings':strings, 'values':values})
        global_mean = data['values'].mean() 
        grp = data.groupby('strings')
        summary = grp.agg(['sum', 'count'])['values']
        for string, row in summary.iterrows():
            tokens = self.get_tokens(string)
            for tok in tokens:
                value_sum[tok] += row['sum']
                tok_count[tok] += row['count']
        
        C = 500.0
        tok_count = pd.Series(tok_count)
        value_sum = pd.Series(value_sum)
        tok_mean = (value_sum + C * global_mean) / (tok_count + C)
        odds = tok_mean - global_mean

        # the random forest to overfit to those.
        odds = odds[tok_count>100]
        self.odds_dict = defaultdict(lambda :0, odds.to_dict())

    def run(self, input_string):
        toks = self.get_tokens(input_string)
        odds = np.sum(map(lambda x:self.odds_dict[x], toks))
        return odds

    def is_fit(self):
        return isinstance(self.odds_dict, dict)

    def save(self, fname):
        config = {'odds_dict': dict(self.odds_dict), 
                  'string_name':self.string_name, 
                  'value_name':self.value_name, 
                  'tok_type': self.tok_type}
        if os.path.isdir(fname):
            file_name = '{}_{}_{}_odds.json'.format(self.value_name, self.string_name, self.tok_type)
            fname = os.path.join(fname, file_name) 
        json.dump(config, open(fname, 'w'))


def save_loan_info(loans):
    f = open(os.path.join(saved_prod_data_dir, 'employer_data.csv'),'a')
    for l in loans:
        f.write('{}|{}|{}\n'.format(l['id'], l['currentJobTitle'],l['currentCompany']))
        l['details_saved']=True
    f.close()

    f = open(os.path.join(saved_prod_data_dir, 'all_api_data.csv'),'a')
    for l in loans:
        save_str = '|'.join(['{}|{}'.format(k,v) for k,v in sorted(l.items())])
        f.write(save_str + '\n')
        l['details_saved']=True
    f.close()

  
class FeatureCalculator(object):
    computed_features = ['credit_length', 'int_pymt', 'revol_bal-loan']
    required_features = ['earliestCrLine', 'loanAmount']

    def __init__(self):
        pass

    def _required_fields_exist(self, data):
        if isinstance(data, dict):
            return all([fld in data.keys() for fld in self.required_features]) 

    def compute(self, data):
        print 'Adding Computed features'
        if isinstance(data, pd.DataFrame):   #historical data
            one_year = 365*24*60*60*1e9
            data['credit_length'] = ((data['issue_d'] - data['earliestCrLine']).astype(int)/one_year)
            data['credit_length'] = data['credit_length'].apply(lambda x: max(-1,round(x,0)))
            data['even_loan_amnt'] = data['loanAmount'].apply(lambda x: float(x==round(x,-3)))
        else:   #API data
            earliest_credit = dt.strptime(data['earliestCrLine'].split('T')[0],'%Y-%m-%d')
            seconds_per_year = 365*24*60*60.0
            data['credit_length'] = (dt.now() - earliest_credit).total_seconds() / seconds_per_year
            data['even_loan_amount'] = float(data['loanAmount'] == np.round(data['loanAmount'],-3))

        data['int_pymt'] = data['loanAmount'] * data['intRate'] / 1200.0
        data['revol_bal-loan'] = data['revolBal'] - data['loanAmount']
        data['pct_med_inc'] = data['annualInc'] / data['census_median_income']
        data['pymt_pct_inc'] = data['installment'] / data['annualInc'] 
        data['revol_bal_pct_inc'] = data['revolBal'] / data['annualInc']
        data['int_pct_inc'] = data['int_pymt'] / data['annualInc'] 
        data['loan_pct_income'] = data['loanAmount'] / data['annualInc']
        data['cur_bal-loan_amnt'] = data['totCurBal'] - data['loanAmount'] 
        data['cur_bal_pct_loan_amnt'] = data['totCurBal'] / data['loanAmount'] 
        data['mort_bal'] = data['totCurBal'] - data['totalBalExMort']
        data['mort_pct_credit_limit'] = data['mort_bal'] * 1.0 / data['totHiCredLim']
        data['mort_pct_cur_bal'] = data['mort_bal'] * 1.0 / data['totCurBal']
        data['revol_bal_pct_cur_bal'] = data['revolBal'] * 1.0 / data['totCurBal']


class FeatureManager(object):
    '''
    FeatureManager is responsible for processing the raw API loan data and fully
    populate the loan dictionary with all fields required by the prepayment and 
    default models
    '''

    def __init__(self):
        self.api_parser = APIDataParser()
        self.location_manager = LocationDataManager()
        self.feature_calculator = FeatureCalculator()

    def process_new_loan(self, loan):
        self.api_parser.parse(loan)
        features = self.location_manager.get_features(loan['addrZip'], loan['addrState'])
        loan.update(features)
        self.feature_calculator(loan)



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
    

def estimate_default_curves():
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
        '''
        plt.figure()
        plt.plot(empirical_default, 'b')
        plt.plot(smoothed_default, 'r')
        plt.title(i)
        plt.grid()
        plt.show()
        '''

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

    default_curves_fname = os.path.join(training_data_dir, 'default_curves.json')
    json.dump(default_curves, open(default_curves_fname, 'w'), indent=4)


def estimate_prepayment_curves():
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
    prepay_curves = dict((str(k),v) for k,v in prepay_curves.items())
    prepay_curves_fname = os.path.join(training_data_dir, 'prepay_curves_rev_util.json')
    json.dump(prepay_curves, open(prepay_curves_fname, 'w'), indent=4)
   

def build_zip3_to_location_names():
    import bls 
    cw = pd.read_csv(os.path.join(reference_data_dir, 'CBSA_FIPS_MSA_crosswalk.csv'))
    grp = cw.groupby('FIPS')
    f2loc = dict([(k,list(df['CBSA Name'].values)) 
                  for k in grp.groups.keys()
                  for df in [grp.get_group(k)]])

    z3f = json.load(open(os.path.join(reference_data_dir, 'zip3_fips.json'),'r'))
    z2loc = dict()
    for z,fips in z3f.items():
        loc_set = set()
        for f in fips:
            if f in f2loc.keys():
                loc_set.update([bls.convert_unicode(loc) for loc in f2loc[f]])
        z2loc[int(z)] = sorted(list(loc_set))
     
    for z in range(1,1000):
        if z not in z2loc.keys() or len(z2loc[z])==0:
            z2loc[z] = ['No location info for {} zip'.format(z)]

    json.dump(z2loc, open(os.path.join(reference_data_dir, 'zip2location.json'),'w'))

def build_zip3_to_primary_city():
    data= pd.read_csv(os.path.join(reference_data_dir, 'zip_code_database.csv'))
    data['place'] = ['{}, {}'.format(c,s) for c,s in zip(data['primary_city'].values, data['state'].values)]

    z2city = defaultdict(lambda :list())
    for z,c in zip(data['zip'].values, data['place'].values):
        z2city[int(z/100)].append(c)

    z2primarycity = dict()
    z2primary2 = dict()
    for z,citylist in z2city.items():
        z2primarycity[z] = Counter(citylist).most_common(1)[0][0]
        z2primary2[z] = Counter(citylist).most_common(2)

    for i in range(0,1000):
        if i not in z2primarycity.keys():
            z2primarycity[i] = 'No primary city for zip3 {}'.format(i)

    json.dump(z2primarycity, open(os.path.join(reference_data_dir, 'zip2primarycity.json'),'w'))


def save_charity_pct():
    irs = pd.read_csv('/Users/marcshivers/Downloads/12zpallagi.csv')
    irs['zip3'] = irs['zipcode'].apply(lambda x:int(x/100))
    irs = irs.ix[irs['AGI_STUB']<5]
    grp = irs.groupby('zip3')
    grp_sum = grp.sum()
    tax_df = pd.DataFrame({'agi':grp_sum['A00100'], 'charity':grp_sum['A19700']})
    tax_df['pct'] = tax_df['charity'] * 1.0 / tax_df['agi']
    json.dump(tax_df['pct'].to_dict(), open(os.path.join(reference_data_dir, 'charity_pct.json'), 'w'))


def get_external_data():
    #def build_zip3_to_hpi():
    z2c = pd.read_csv(os.path.join(reference_data_dir, 'zip2cbsa.csv'))
    z2c['zip3'] = z2c['ZIP'].apply(lambda x: int(x/100))
    z2c = z2c[z2c['CBSA']<99999]
    grp = z2c.groupby('zip3')

    z2clist = dict()
    for z3 in grp.groups.keys():
        g = grp.get_group(z3)
        z2clist[z3] = sorted(list(set(g['CBSA'].values)))


    # get metro hpi for main areas
    link = "http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_metro.csv"
    cols = ['Location','CBSA', 'yr','qtr','index', 'stdev']
    try:
        data = pd.read_csv(link, header=None, names=cols)
        data.to_csv(os.path.join(fhfa_data_dir, 'HPI_AT_metro.csv'))
    except:
        print 'Failed to read FHFA website HPI data; using cached data'
        data = pd.read_csv(os.path.join(fhfa_data_dir,'HPI_AT_metro.csv'), header=None, names=cols)

    data = data[data['index']!='-']
    data['index'] = data['index'].astype(float)
    data['yyyyqq'] = 100 * data['yr'] + data['qtr'] 
    data = data[data['yyyyqq']>199000]

    index = np.log(data.pivot('yyyyqq', 'CBSA', 'index'))
    hpa1q = index - index.shift(1) 
    hpa1y = index - index.shift(4) 
    hpa5y = index - index.shift(20)
    hpa10y = index - index.shift(40)

    hpa1 = dict()
    hpa4 = dict()
    hpa20 = dict()
    hpa40 = dict()
    for z,c in z2clist.items():
        hpa1[z] = hpa1q.ix[:,c].mean(1)
        hpa4[z] = hpa1y.ix[:,c].mean(1)
        hpa20[z] = hpa5y.ix[:,c].mean(1)
        hpa40[z] = hpa10y.ix[:,c].mean(1)
    hpa1 = pd.DataFrame(hpa1).dropna(axis=1, how='all')
    hpa4 = pd.DataFrame(hpa4).dropna(axis=1, how='all')
    hpa20 = pd.DataFrame(hpa20).dropna(axis=1, how='all')
    hpa40 = pd.DataFrame(hpa40).dropna(axis=1, how='all')

    hpa1.to_csv(os.path.join(fhfa_data_dir,'hpa1.csv'))
    hpa4.to_csv(os.path.join(fhfa_data_dir,'hpa4.csv'))
    hpa20.to_csv(os.path.join(fhfa_data_dir,'hpa20.csv'))
    hpa40.to_csv(os.path.join(fhfa_data_dir,'hpa40.csv'))

    '''
    # get non-metro hpi, for other zip codes
    link='http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_nonmetro.xls'
    try:
        data = pd.read_excel(link, skiprows=2)
        data.to_csv(os.path.join(p.parent_dir, 'HPI_AT_nonmetro.csv'))
    except:
        data = pd.read_csv(os.path.join(p.parent_dir,'HPI_AT_nonmetro.csv'))

    grp = data.groupby('State')
    tail5 = grp.tail(21).groupby('State')['Index']
    chg5 = np.log(tail5.last()) - np.log(tail5.first())
    tail1 = grp.tail(5).groupby('State')['Index']
    chg1 = np.log(tail1.last()) - np.log(tail1.first())
    chg = 100.0 * pd.DataFrame({'1yr':chg1, '5yr':chg5})

    return chg
    '''

    # downloads the monthly non-seasonally adjusted employment data, and saves csv files for
    # monthly labor force size, and number of unemployed by fips county code, to use to construct
    # historical employment statistics by zip code for model fitting
    z2f = json.load(file(os.path.join(reference_data_dir, 'zip3_fips.json'),'r'))

    #z2f values are lists themselves; this flattens it
    all_fips = []
    for f in z2f.values():
        all_fips.extend(f) 
    fips_str = ['0'*(5-len(str(f))) + str(f) for f in all_fips]

    data_code = dict()
    data_code['03'] = 'unemployment_rate'
    data_code['04'] = 'unemployment'
    data_code['05'] = 'employment'
    data_code['06'] = 'labor force'

    #series_id = 'CN{}00000000{}'.format(fips, '06') 
    cols = ['series_id', 'year', 'period', 'value']
    link1 = 'http://download.bls.gov/pub/time.series/la/la.data.0.CurrentU10-14'
    link2 = 'http://download.bls.gov/pub/time.series/la/la.data.0.CurrentU15-19'
    cvt = dict([('series_id', lambda x: str(x).strip()) ])
    data1 = pd.read_csv(link1, delimiter=r'\s+', usecols=cols, converters=cvt)
    data2 = pd.read_csv(link2, delimiter=r'\s+', usecols=cols, converters=cvt)
    data = pd.concat([data1, data2], ignore_index=True)
    data = data.replace('-', np.nan)
    data = data.dropna()
    data = data.ix[data['period']!='M13']
    data['value'] = data['value'].astype(float)
    data['yyyymm'] = 100 * data['year'] + data['period'].apply(lambda x: int(x[1:]))
    data['fips'] = [int(f[5:10]) for f in data['series_id']]
    data['measure'] = [f[-2:] for f in data['series_id']]
    data['region'] = [f[3:5] for f in data['series_id']]
    del data['year'], data['period'], data['series_id']
    county_data = data.ix[data['region']=='CN']
    labor_force = county_data[county_data['measure']=='06'][['fips','yyyymm','value']]
    labor_force = labor_force.pivot('yyyymm','fips','value')
    unemployed = county_data[county_data['measure']=='04'][['fips','yyyymm','value']]
    unemployed = unemployed.pivot('yyyymm','fips','value')
    labor_force.to_csv(os.path.join(bls_data_dir, 'labor_force.csv'))
    unemployed.to_csv(os.path.join(bls_data_dir, 'unemployed.csv'))

    # reads the monthly labor force size, and number of unemployed by fips county code,
    # and constructs historical employment statistics by zip code for model fitting
    labor_force = labor_force.fillna(0).astype(int).rename(columns=lambda x:int(x))
    unemployed = unemployed.fillna(0).astype(int).rename(columns=lambda x:int(x))

    urates = dict()
    for z,fips in z2f.items():
        ue = unemployed.ix[:,fips].sum(1)
        lf = labor_force.ix[:,fips].sum(1)
        ur = ue/lf
        ur[lf==0]=np.nan
        urates[z] = ur

    urate = pd.DataFrame(urates)
    urate.to_csv(os.path.join(bls_data_dir, 'urate_by_3zip.csv'))
        
