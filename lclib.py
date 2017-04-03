import os
import json
import urllib
import requests
import smtplib
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from collections import defaultdict, Counter
import scipy.signal
from matplotlib import pyplot as plt
import personalized as p

LARGE_INT = 9999 
NEGATIVE_INT = -1 

data_dir = os.path.join(p.parent_dir, 'data')
loanstats_dir = os.path.join(p.parent_dir, 'data/loanstats')
training_data_dir = os.path.join(p.parent_dir, 'data/training_data')
reference_data_dir = os.path.join(p.parent_dir, 'data/reference_data')
bls_data_dir = os.path.join(p.parent_dir, 'data/bls_data')
fhfa_data_dir = os.path.join(p.parent_dir, 'data/fhfa_data')
saved_prod_data_dir = os.path.join(p.parent_dir, 'data/saved_prod_data')

payments_file = os.path.join(loanstats_dir, 'PMTHIST_ALL_20170315.csv')
cached_training_data_file = os.path.join(training_data_dir, 'cached_training_data.csv')

update_hrs = [1,5,9,13,17,21]

# clean dataframe
class StringConverter(object):
    def __init__(self):
        self.accepted_fields = ['homeOwnership',
                                'purpose',
                                'grade',
                                'subGrade',
                                'isIncV', 
                                'isIncVJoint',
                                'initialListStatus',
                                'empLength',
                                'addrZip',
                                'empTitle']
        self.home_map = dict([('ANY', 0), ('NONE',0), ('OTHER',0), 
                              ('RENT',1), ('MORTGAGE',2), ('OWN',3)])
        self.purpose_dict = defaultdict(lambda :np.nan)
        self.purpose_dict.update([('credit_card', 0), ('credit_card_refinancing', 0), 
                                  ('debt_consolidation',1), 
                                  ('home_improvement',2), 
                                  ('car',3), ('car_financing',3), 
                                  ('educational',4), 
                                  ('house',5), ('home_buying',5),
                                  ('major_purchase',6), 
                                  ('medical_expenses',7), ('medical',7), 
                                  ('moving',8), ('moving_and_relocation',8), 
                                  ('other',9),
                                  ('renewable_energy',10), ('green_loan',10),
                                  ('business',11),('small_business',11),
                                  ('vacation',12), 
                                  ('wedding',13)])
        grades = list('ABCDEFG')
        self.grade_map = defaultdict(lambda :np.nan, zip(grades, range(len(grades))))
        subgrades = ['{}{}'.format(l,n) for l in 'ABCDEFG' for n in range(1,6)]
        self.subgrade_map = defaultdict(lambda :np.nan, zip(subgrades, range(len(subgrades))))
        loanstats_verification_dict = dict([('Verified',2), ('Source Verified',1), ('Not Verified',0)]) 
        api_verification_dict = dict([('VERIFIED',2), ('SOURCE_VERIFIED',1), ('NOT_VERIFIED',0)])
        self.income_verification = loanstats_verification_dict
        self.income_verification.update(api_verification_dict)
        self.init_status_dict = dict([('f',0), ('F',0), ('w',1), ('W',1)])

    def _employment_length_converter(self, el):
        el=el.replace('< 1 year', '0')
        el=el.replace('1 year','1')
        el=el.replace('10+ years', '11')
        el=el.replace('n/a', '-1')
        el=el.replace(' years', '')
        return int(el)
    
    def _convert_grade(self, value):
        return self.grade_map[value]

    def _convert_homeOwnership(self, value):
        return self.home_map[value.upper()]

    def _convert_purpose(self, value):
        value = value.lower().replace(' ', '_')
        return self.purpose_dict[value]

    def _convert_subGrade(self, value):
        return self.subgrade_map[value]

    def _convert_inc_verification(self, value):
        return self.income_verification[value]

    def _convert_initialListStatus(self, value):
        return self.init_status_dict[value]

    def _convert_empLength(self, value):
        return self._employment_length_converter(value)

    def _convert_addrZip(self, value):
        return int(value[:3])

    def _convert_empTitle(self, value):
        return only_ascii(field)
         
    def convert(self, field, value):
        if field == 'homeOwnership':
            if value.upper() in self.home_map.keys():
                return self.home_map[value.upper()]
        elif field == 'purpose':
            value = value.lower().replace(' ', '_')
            if value in self.purpose_dict.keys():
                return self.purpose_dict[value]
        elif field == 'grade':
            if value in self.grade_map.keys():
                return self.grade_map[value]
        elif field == 'subGrade':
            if value in self.subgrade_map.keys():
                return self.subgrade_map[value]
        elif field in ['isIncV', 'isIncVJoint']:
            if value in self.income_verification.keys():
                return self.income_verification[value]
        elif field == 'initialListStatus':
            if value in self.init_status_dict.keys():
                return self.init_status_dict[value]
        elif field == 'empLength':
            return self._employment_length_converter(value)
        elif field == 'addrZip':
            return int(value[:3])
        elif field == 'empTitle':
            return '^{}$'.format(only_ascii(value))
        else:
            return value

def substrings(x):
    toks = list()
    x = '^{}$'.format(x)
    #add all whole words
    toks.extend(x.split())
    toks.extend([x[i:i+k] for k in range(1,len(x)+1) 
        for i in range(max(1,len(x)-k+1))])
    return list(set(toks))

def clean_title(x):
    x = str(x).strip().lower()
    x = x.replace("'","")
    x = x.replace('"','')
    x = x.replace('/', ' ')
    for tok in '`~!@#$%^&*()_-+=\|]}[{;:/?.>,<':
        x = x.replace(tok,'_')
    x = '^{}$'.format(x) 
    return x

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

def get_loanstats2api_map():
    col_name_file = open(os.path.join(reference_data_dir, 'loanstats2api.txt'), 'r')
    col_name_map = dict([line.strip().split(',') for line in col_name_file.readlines()])
    return col_name_map

def cache_training_data():
    #rename columns to match API fields
    col_name_map = get_loanstats2api_map()

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

    parser = APIDataParser()
    for field in parser.null_fill_fields():
        if field in df.columns:
            fill_value = parser.null_fill_value(field)
            print 'Filling {} nulls with {}'.format(field, fill_value)
            df[field] = df[field].fillna(fill_value)

    string_converter = StringConverter()
    for col in string_converter.accepted_fields:
        if col in df.columns:
            print 'Converting {} string to numeric'.format(col)
            func = np.vectorize(lambda x: string_converter.convert(col, x))
            df[col] = df[col].apply(func)
   
    df['clean_title'] = df['empTitle'].apply(clean_title)
    df['title_capitalization'] = df['empTitle'].apply(tokenize_capitalization)

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

    # add title rank feature 
    print 'Adding Title Ranks'
    clean_title_count = Counter(df.ix[df.in_sample, 'clean_title'].values)
    clean_titles_sorted = [ttl[0] for ttl in sorted(clean_title_count.items(), key=lambda x:-x[1])]
    clean_title_rank_dict = dict(zip(clean_titles_sorted, range(len(clean_titles_sorted))))
    json.dump(clean_title_rank_dict, open(os.path.join(training_data_dir, 'clean_title_rank_map.json'),'w'))
    clean_title_rank_map = defaultdict(lambda :1e9, clean_title_rank_dict)
    df['clean_title_rank'] = df['clean_title'].apply(lambda x:clean_title_rank_map[x])

    # process job title features
    sample = (df.in_sample)
    add_log_odds_feature(df, sample, string_fld='empTitle', numeric_fld='12m_prepay', tok='alltoks')
    #add_log_odds_feature(df, sample, string_fld='clean_title', numeric_fld='12m_wgt_default', tok=4)
    #add_log_odds_feature(df, sample, string_fld='title_capitalization', numeric_fld='12m_wgt_default', tok=4)
    #add_log_odds_feature(df, sample, string_fld='empTitle', numeric_fld='12m_wgt_default', tok='word')
    #add_log_odds_feature(df, sample, string_fld='empTitle', numeric_fld='12m_wgt_default', tok='phrase')

    sample = (df.grade>=2) & (df.in_sample)
    add_log_odds_feature(df, sample, string_fld='empTitle', numeric_fld='12m_wgt_default', tok='alltoks')
    #add_log_odds_feature(df, sample, string_fld='clean_title', numeric_fld='12m_prepay', tok=4)
    #add_log_odds_feature(df, sample, string_fld='empTitle', numeric_fld='12m_prepay', tok='word')

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
    df['max_urate'] = [ur[a][:b].max() for a,b in zip(df['addrZip'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['min_urate'] = [ur[a][:b].min() for a,b in zip(df['addrZip'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['urate_range'] = df['max_urate'] - df['min_urate'] 

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
    z2mi = json.load(open(os.path.join(reference_data_dir, 'zip2median_inc.json'),'r'))
    z2mi = dict([(int(z), float(v)) for z,v in zip(z2mi.keys(), z2mi.values())])
    z2mi = defaultdict(lambda :np.mean(z2mi.values()), z2mi)
    df['med_inc'] = df['addrZip'].apply(lambda x:z2mi[x])

    # Add calculated features 
    print 'Adding Computed features'
    one_year = 365*24*60*60*1e9
    df['credit_length'] = ((df['issue_d'] - df['earliestCrLine']).astype(int)/one_year)
    df['credit_length'] = df['credit_length'].apply(lambda x: max(-1,round(x,0)))
    df['even_loan_amnt'] = df['loanAmount'].apply(lambda x: float(x==round(x,-3)))
    df['int_pymt'] = df['loanAmount'] * df['intRate'] / 1200.0
    df['revol_bal-loan'] = df['revolBal'] - df['loanAmount']
    df['pct_med_inc'] = df['annualInc'] / df['med_inc']
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

    df.to_csv(cached_training_data_file)


def only_ascii(s):
    return ''.join([c for c in s if ord(c)<128])

def get_tokens(x, tok_type):
    if len(x)==0:
        return [] 
    if tok_type=='word':
        x = x.replace('^','').replace('$','')
        toks = x.split()
    elif tok_type=='phrase':
        toks = [x]
    elif tok_type=='alltoks':
        toks = list()
        for i in range(1, len(x)+1):
            toks.extend(get_tokens(x, tok_type=i))
    elif isinstance(tok_type, int):
        toks = [x[i:i+tok_type] for i in range(max(1,len(x)-tok_type+1))]
    else:
        raise Exception('unknown tok_type')
    return toks

def calc_log_odds(x, log_odds_dict, tok_type):
    toks = get_tokens(x, tok_type)
    log_odds = np.sum(map(lambda x:log_odds_dict[x], toks))
    return log_odds

def tokenize_capitalization(txt):
    txt = txt.strip()
    #replace upper characters with 'C', lower with 'c', etc...
    tokstr = []
    for c in txt:
        if c.isupper():
            tokstr.append('C')
        elif c.islower():
            tokstr.append('c')
        elif c.isdigit():
            tokstr.append('n')
        elif c.isspace():
            tokstr.append(' ')
        else:
            tokstr.append('p') #punctuation
    tokenized = ''.join(tokstr)
    tokenized = '^{}$'.format(tokenized) #add leading a trailing token to distinguish first and last characters.
    return tokenized

def fast_create_log_odds(df, string_fld='clean_title', numeric_fld='12m_wgt_default', tok_type=4):
    ''' creates a log odds dictionary.  The tok_type field is either the length of the string token
    to use, or if tok_type='word', the tokens are entire words'''
    fld_sum = defaultdict(lambda :0)
    fld_count = defaultdict(lambda :0)
    g = df[[string_fld, numeric_fld]].groupby(string_fld)
    data = g.agg(['sum', 'count'])[numeric_fld]
    numeric_mean = df[numeric_fld].mean() 
    for title, row in data.iterrows():
        tokens = get_tokens(title, tok_type)
        for tok in tokens:
            fld_sum[tok] += row['sum']
            fld_count[tok] += row['count']
    
    C = 500.0
    fld_count = pd.Series(fld_count)
    fld_sum = pd.Series(fld_sum)
    logodds = (fld_sum + C * numeric_mean) / (fld_count + C)
    logodds = np.log(logodds) - np.log(numeric_mean)

    # we need to remove the items with only small counts, since these would cause
    # the random forest to overfit to those.
    logodds = logodds[fld_count>100]

    return logodds.to_dict() 

def add_log_odds_feature(df, sample, string_fld, numeric_fld, tok): 
    if 'prepay' in numeric_fld:
        predictor = 'prepay'
    elif 'default' in numeric_fld:
        predictor = 'default'
    else:
        predictor = 'unknown'

    if isinstance(tok, int):
        tok_str = 'tok{}'.format(tok)
    else: 
        tok_str = tok

    name = '{}_{}_{}_odds'.format(predictor, string_fld, tok_str)
    print 'Calculating {} {} {} odds'.format(string_fld, tok_str, predictor)
    lo_dict = fast_create_log_odds(df.ix[sample], string_fld=string_fld,
                                   numeric_fld=numeric_fld, tok_type=tok)
    json.dump(lo_dict, open(os.path.join(training_data_dir, '{}.json'.format(name)),'w'))
    lo = defaultdict(lambda :0, lo_dict)
    odds_map = lambda x: calc_log_odds(x, lo, tok)
    df[name] = df[string_fld].apply(odds_map)


def reset_time():
    try:
        import os
        print 'Attempting to adjust system time'
        response = os.popen('sudo ntpdate -u time.apple.com')
        print response.read()
        print 'Reset Time Successfully!'  
    except:
        print 'Failed to reset system time' 

def get_web_time():
    try:
        import os
        print 'Attempting to adjust system time'
        response = os.popen('sudo ntpdate -u time.apple.com')
        print response.read()
        print 'Success!'
        response = urllib.urlopen('http://www.timeapi.org/est/now').read()
        time = dt.strptime(response[:-6], '%Y-%m-%dT%H:%M:%S') 
        print 'Web Time: {}'.format(time)
        print 'Sys Time: {}'.format(dt.now())
    except:
        print 'Failed to get web time'
        time = dt.now()

    return time 


def hourfrac(tm):
    return (tm.hour + tm.minute/60.0 + tm.second / 3600.0)


def invest_amount(loan, min_irr, max_invest=None):
    if max_invest==None:
        max_invest = 500
    if loan['stress_irr'] < min_irr:
        stage_amount = 0 
    else:
        # invest $25 for every 25bps that stress_irr exceeds min_irr
        stage_amount =  max(0, 25 * np.ceil(400*(loan['stress_irr'] - min_irr)))
    #don't invest in grade G loans; model doesn't work as well for those
    if loan['grade'] >= 'G':
        stage_amount = 0
    loan['max_stage_amount'] =  min(max_invest, stage_amount) 


def sleep_seconds(win_len=30):
     # win_len is the number of seconds to continuously check for new loans.
     # The period ends at the official update time.  
     now = dt.now()
     tm = now + td(seconds=win_len/2.0)
     update_seconds = np.array([60*60*(hr - hourfrac(tm)) for hr in update_hrs])
     center = min(abs(update_seconds))
     max_sleep_seconds = 0.8 * max(center - win_len, 0)
     #seconds_to_hour = (59 - now.minute) * 60.0 + (60 - now.second)
     return max_sleep_seconds


def detail_str(loan):

    pstr = 'BaseIRR: {:1.2f}%'.format(100*loan['base_irr'])
    pstr += ' | StressIRR: {:1.2f}%'.format(100*loan['stress_irr'])
    pstr += ' | BaseIRRTax: {:1.2f}%'.format(100*loan['base_irr_tax'])
    pstr += ' | StressIRRTax: {:1.2f}%'.format(100*loan['stress_irr_tax'])
    pstr += ' | IntRate: {}%'.format(loan['intRate'])

    pstr += '\nDefaultRisk: {:1.2f}%'.format(100*loan['default_risk'])
    pstr += ' | DefaultMax: {:1.2f}%'.format(100*loan['default_max'])
    pstr += ' | PrepayRisk: {:1.2f}%'.format(100*loan['prepay_risk'])
    pstr += ' | PrepayMax: {:1.2f}%'.format(100*loan['prepay_max'])
    pstr += ' | RiskFactor: {:1.2f}'.format(loan['risk_factor'])

    pstr += '\nInitStatus: {}'.format(loan['initialListStatus'])
    pstr += ' | Staged: ${}'.format(loan['staged_amount'])

    pstr += '\nLoanAmnt: ${:1,.0f}'.format(loan['loanAmount'])
    pstr += ' | Term: {}'.format(loan['term'])
    pstr += ' | Grade: {}'.format(loan['subGrade'])
    pstr += ' | Purpose: {}'.format(loan['purpose'])
    pstr += ' | LoanId: {}'.format(loan['id'])
    
    pstr += '\nRevBal: ${:1,.0f}'.format(loan['revolBal'])
    pstr += ' | RevUtil: {}%'.format(loan['revolUtil'])
    pstr += ' | DTI: {}%'.format(loan['dti'])
    pstr += ' | Inq6m: {}'.format(loan['inqLast6Mths'])
    pstr += ' | 1stCredit: {}'.format(loan['earliestCrLine'].split('T')[0])
    pstr += ' | fico: {}'.format(loan['ficoRangeLow'])
  
    pstr += '\nJobTitle: {}'.format(loan['currentJobTitle'])
    pstr += ' | Company: {}'.format(loan['currentCompany'])
    
    pstr += '\nClean Title Log Odds: {:1.2f}'.format(loan['clean_title_log_odds'])
    pstr += ' | Capitalization Log Odds: {:1.2f}'.format(loan['capitalization_log_odds'])
    pstr += ' | Income: ${:1,.0f}'.format(loan['annualInc'])
    pstr += ' | Tenure: {}'.format(loan['empLength'])

    pstr += '\nLoc: {},{}'.format(loan['addrZip'], loan['addrState'])
    pstr += ' | MedInc: ${:1,.0f}'.format(loan['med_income'])
    pstr += ' | URate: {:1.1f}%'.format(100*loan['urate'])
    pstr += ' | 12mChg: {:1.1f}%'.format(100*loan['urate_chg'])

    pstr += '\nHomeOwn: {}'.format(loan['homeOwnership'])
    pstr += ' | PrimaryCity: {}'.format(loan['primaryCity'])
    pstr += ' | HPA1: {:1.1f}%'.format(loan['HPA1Yr'])
    pstr += ' | HPA5: {:1.1f}%'.format(loan['HPA5Yr'])

    return pstr 

email_keys = ['accOpenPast24Mths','mthsSinceLastDelinq', 'mthsSinceRecentBc', 'bcUtil', 'totCollAmt', 
        'isIncV', 'numTlOpPast12m', 'totalRevHiLim', 'mthsSinceRecentRevolDelinq', 'revolBal',
        'pubRec', 'delinq2Yrs', 'inqLast6Mths', 'numOpRevTl', 'pubRecBankruptcies', 'numActvRevTl',
        'mthsSinceRecentBcDlq', 'revolUtil', 'numIlTl', 'numRevTlBalGt0', 'numTl90gDpd24m', 'expDefaultRate', 
        'initialListStatus', 'moSinOldIlAcct', 'numBcTl', 'totHiCredLim', 'delinqAmnt', 'moSinOldRevTlOp', 
        'numRevAccts', 'totalAcc', 'mortAcc', 'mthsSinceRecentInq', 'moSinRcntRevTlOp','totCurBal', 
        'collections12MthsExMed', 'dti', 'numActvBcTl', 'pctTlNvrDlq', 'totalBcLimit',
        'accNowDelinq', 'numTl30dpd', 'percentBcGt75', 'numBcSats', 'openAcc', 'numAcctsEver120Ppd', 'bcOpenToBuy',
        'numTl120dpd2m', 'taxLiens', 'mthsSinceLastRecord', 'totalBalExMort', 'avgCurBal', 'moSinRcntTl', 
        'mthsSinceLastMajorDerog', 'totalIlHighCreditLimit', 'chargeoffWithin12Mths', 'clean_title_rank']

email_keys = [ 'isIncV', 'totalRevHiLim', 'revolBal', 'numRevTlBalGt0', 'numTl90gDpd24m',
        'initialListStatus', 'totHiCredLim', 'delinqAmnt', 'mortAcc', 'totCurBal', 
        'pctTlNvrDlq', 'totalBcLimit', 'numTl30dpd', 'percentBcGt75', 'numAcctsEver120Ppd',
        'numTl120dpd2m', 'totalBalExMort', 'avgCurBal',
        'totalIlHighCreditLimit', 'clean_title_rank']

def all_detail_str(loan):
    all_details = '\n'.join(sorted(['{}: {}'.format(k,v) for k,v in loan.items() 
        if k in email_keys or k.startswith('dflt')]))
    return all_details 


def send_email(msg):
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(p.email, p.smtp_password)
    msg_list = ['From: {}'.format(p.email), 
                'To: {}'.format(p.email), 
                'Subject: Lending Club', '']
    msg_list.append(msg)
    msg = '\r\n'.join(msg_list) 
    server.sendmail(p.email, [p.email], msg)
    return


def email_details(acct, loans, info):
    msg_list = list()
    msg_list.append('{} orders have been staged to {}'.format(len(loans), acct))
    msg_list.append('{} total loans found, valued at ${:1,.0f}'.format(info['num_new_loans'], info['value_new_loans']))

    count_by_grade = dict(zip('ABCDEFG', np.zeros(7)))
    for loan in loans:
        if loan['email_details'] == True:
            count_by_grade[loan['grade']] += 1

    g = info['irr_df'].groupby(['grade', 'initialListStatus'])
    def compute_metrics(x):
        result = {'irr_count': x['base_irr'].count(), 'irr_mean': x['base_irr'].mean()}
        return pd.Series(result, name='metrics')
    msg_list.append(g.apply(compute_metrics).to_string())

    irr_msgs = list()
    irr_msgs.append('Average IRR is {:1.2f}%.'.format(100*info['average_irr']))
    for grade in sorted(info['irr_by_grade'].keys()):
        avg = 100*np.mean(info['irr_by_grade'][grade])
        num = len(info['irr_by_grade'][grade])
        bought = int(count_by_grade[grade])
        irr_msgs.append('Average of {} grade {} IRRs is {:1.2f}%; {} staged.'.format(num, grade, avg, bought))
    msg_list.append('\r\n'.join(irr_msgs))

    for loan in loans:
        if loan['email_details'] == True:
            msg_list.append(detail_str(loan))
            msg_list.append(all_detail_str(loan))
        loan['email_details'] = False
    msg_list.append('https://www.lendingclub.com/account/gotoLogin.action')
    msg_list.append('Send at MacOSX clocktime {}'.format(dt.now()))
    msg = '\r\n\n'.join(msg_list) 
    send_email(msg)
    return


def load_census_data():
    # loads the census bureaus low-income area data
    fname = os.path.join(reference_data_dir, 'lya2014.txt')
    rows = open(fname,'r').read().split('\r\n')
    data = [r.split() for r in rows]
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.dropna()
    df['STATE'] = df['STATE'].astype(int)
    df['CNTY'] = df['CNTY'].astype(int)
    df['FIPS'] = 1000 * df['STATE'] + df['CNTY']
    df['LYA'] = df['LYA'].astype(float)
    df['CENINC'] = df['CENINC'].astype(float)
    df = df[df['LYA'] <= 1]  # remove missing data (LYA==9)
    return df

def load_nonmetro_housing():
    link='http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_nonmetro.xls'
    try:
        data = pd.read_excel(link, skiprows=2)
        data.to_csv(os.path.join(fhfa_data_dir, 'HPI_AT_nonmetro.csv'))
    except:
        data = pd.read_csv(os.path.join(fhfa_data_dir,'HPI_AT_nonmetro.csv'))
        print '{}: Failed to load FHFA nonmetro data; using cache\n'.format(dt.now())

    grp = data.groupby('State')
    tail5 = grp.tail(21).groupby('State')['Index']
    chg5 = np.log(tail5.last()) - np.log(tail5.first())
    tail1 = grp.tail(5).groupby('State')['Index']
    chg1 = np.log(tail1.last()) - np.log(tail1.first())
    chg = 100.0 * pd.DataFrame({'1yr':chg1, '5yr':chg5})
    return chg

 
def load_metro_housing():
    # loads the fhfa home price quarterly index data for Census Bureau
    # Statistical Areas
    link = "http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_metro.csv"
    cols = ['Location','CBSA', 'yr','qtr','index', 'stdev']
    try:
        data = pd.read_csv(link, header=None, names=cols)
        data.to_csv(os.path.join(fhfa_data_dir, 'HPI_AT_metro.csv'))
    except:
        data = pd.read_csv(os.path.join(fhfa_data_dir,'HPI_AT_metro.csv'), skiprows=1, header=None, names=cols)
        print '{}: Failed to load FHFA metro data; using cache\n'.format(dt.now())
    data = data[data['index']!='-']
    data['index'] = data['index'].astype(float)
    grp = data.groupby('CBSA')[['Location','CBSA', 'yr','qtr','index', 'stdev']]
    tail5 = grp.tail(21).groupby(['CBSA','Location'])['index']
    chg5 = np.log(tail5.last()) - np.log(tail5.first())
    tail1 = grp.tail(5).groupby(['CBSA','Location'])['index']
    chg1 = np.log(tail1.last()) - np.log(tail1.first())
    chg = 100.0 * pd.DataFrame({'1yr':chg1, '5yr':chg5})
    chg = chg.reset_index(1)
    return chg

    
# Get unemployment rates by County (one-year averages)
def load_bls():

    z2f = load_z2f()

    bls_fname = os.path.join(bls_data_dir, 'bls_summary.csv')
    if os.path.exists(bls_fname):
        update_dt = dt.fromtimestamp(os.path.getmtime(bls_fname))
        days_old = (dt.now() - update_dt).days 
        summary = pd.read_csv(bls_fname)
    else:
        days_old = 999

    if days_old > 14:
        try:
            link = 'http://www.bls.gov/lau/laucntycur14.txt'
            link = 'http://www.bls.gov/web/metro/laucntycur14.txt'
            cols = ['Code', 'StateFIPS', 'CountyFIPS', 'County', 
                'Period', 'CLF', 'Employed', 'Unemployed', 'Rate']
            file = requests.get(link)
            rows = [l.split('|') for l in file.text.split('\r\n') if l.startswith(' CN')]
            data =pd.DataFrame(rows, columns=cols)
            data['Period'] = data['Period'].apply(lambda x:dt.strptime(x.strip()[:6],'%b-%y'))

            # keep only most recent 12 months; note np.unique also sorts
            min_date = np.unique(data['Period'])[1]
            data = data[data['Period']>=min_date]

            # reduce Code to just state/county fips number
            to_float = lambda x: float(str(x).replace(',',''))
            data['FIPS'] = data['Code'].apply(lambda x: int(x.strip()[2:7]))

            # convert numerical data to floats
            for col in ['CLF', 'Unemployed']:
                data[col] = data[col].apply(to_float)
            data = data.ix[:,['Period','FIPS','CLF','Unemployed']]
            lf = data.pivot('Period', 'FIPS','CLF')
            ue = data.pivot('Period', 'FIPS', 'Unemployed')
            
            avg_ur = dict()
            ur = dict()
            ur_chg = dict()
            ur_range = dict()
            for z, fips in z2f.items():
                avg_ur[z] =  ue.ix[1:,fips].sum(1).sum(0) / lf.ix[1:,fips].sum(1).sum(0)
                ur[z] =  ue.ix[-1,fips].sum(0) / lf.ix[-1,fips].sum(0)
                last_year_ur =  ue.ix[1,fips].sum(0) / lf.ix[1,fips].sum(0)
                ur_chg[z] = ur[z] - last_year_ur
                monthly_ue = ue.ix[:, fips].sum(1)
                ur_range[z] = (monthly_ue.max() - monthly_ue.min()) / lf.ix[-1,fips].sum(0) 

            summary = pd.DataFrame({'avg':pd.Series(avg_ur),'current':pd.Series(ur),
                'chg12m':pd.Series(ur_chg), 'ur_range':pd.Series(ur_range)})
            
            summary.to_csv(bls_fname)

        except:
            print '{}: Failed to load BLS laucntycur14 data; using summary cache\n'.format(dt.now())
    
    return summary 
    

def load_z2c():
    ''' Core crosswalk file from www.huduser.org/portal/datasets/usps_crosswalk.htlm'''
    data = pd.read_csv(os.path.join(reference_data_dir, 'z2c.csv'))
    data['3zip'] = (data['ZIP']/100).astype(int)
    data['CBSA'] = data['CBSA'].astype(int)
    data = data[data['CBSA']!=99999]
    grp = data.groupby('3zip')
    z2c = defaultdict(lambda :list())
    for z in grp.groups.keys():
        z2c[z] = list(set(grp.get_group(z)['CBSA'].values))
    return z2c 


def load_z2loc():
    ''' loads a dictionary mapping the first 3 digits of the zip code to a 
    list of location names''' 
    data = json.load(open(os.path.join(reference_data_dir, 'zip2location.json'), 'r'))
    return dict([int(k), locs] for k, locs in data.items())


def load_z2primarycity():
    ''' loads a dictionary mapping the first 3 digits of the zip code to the
    most common city with that zip code prefix'''
    data = json.load(open(os.path.join(reference_data_dir,'zip2primarycity.json')))
    return dict([int(k), locs] for k, locs in data.items())


def load_z2f():
    ''' loads a dictionary mapping the first 3 digits of the zip code to a 
    list of FIPS codes'''
    d = json.load(open(os.path.join(reference_data_dir, 'z2f.json'), 'r'))
    return dict([int(k), [int(v) for v in vals]] for k, vals in d.items())


def load_f2c():
    ''' loads a dictionary mapping FIPS codes to CBSA codes and names '''
    #need a mapping that lists NY-NJ to 35614, rather than 36544
    d = json.load(open(os.path.join(reference_data_dir, 'fips2cbsa.json'), 'r'))
    return dict([int(k), [int(v[0]), v[1]]] for k, v in d.items())


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

    

def construct_z2c(z2f, f2c):
    z2c = dict()
    for z, fips in z2f.iteritems():
        clist = set()
        for f in fips:
            if f in f2c.keys():
               clist.add(f2c[f][0])
        z2c[z] = clist 
    return z2c 


def get_loan_details(lc, loan_id):
    '''
    Returns the loan details, including location, current job title, 
    employer, relisted status, and number of inquiries.
    '''
    payload = {
        'loan_id': loan_id
    }

    # Make request
    try:
        response = lc.session.post('/browse/loanDetailAj.action', data=payload)
        detail = response.json()
        return detail 
    except:
        return -1


def get_urate(bls, zip):
    if zip in bls.index:
        urate = bls.ix[zip,'current']
        ur_chg = bls.ix[zip,'chg12m']
        avg_ur = bls.ix[zip,'avg']
        ur_range = bls.ix[zip,'ur_range']
    else:
        near_zip = bls.index[np.argmin(np.abs(np.array(bls.index)-zip))]
        urate = bls.ix[near_zip,'current'].mean()
        ur_chg = bls.ix[near_zip,'chg12m'].mean()
        avg_ur = bls.ix[near_zip,'avg'].mean()
        ur_range = bls.ix[near_zip,'ur_range'].mean()
    return urate, avg_ur, ur_chg, ur_range 


def get_income_data(census, fips_list):
    # Returns the percentage of census tracts with incomes below $5000/mth, and
    # the median census tract income for the input FIPS codes
    # if the census tract doesn't exist (for overseas military, for example), 
    # return the average for the country.
    df = census[census.FIPS.isin(fips_list)]
    if len(df) > 0:
        result = df.CENINC.median()
    else:
        result = census.CENINC.median()
    
    return result


class APIDataParser(object):

    def __init__(self):
        self.api_fields = get_loanstats2api_map().values()
        self.string_converter = StringConverter()
        self.ok_to_be_null = ['dtiJoint',
                              'desc',
                              'isIncVJoint',
                              'investorCount',
                              'annualIncJoint',
                              ]

    def null_fill_value(self, field):
        if( field.startswith('mthsSinceLast')
                or field.startswith('mthsSinceRecent')
                or field.startswith('moSinRcnt')):
            return LARGE_INT
        elif (field.startswith('moSinOld')
                or field.startswith('num')
                or field.endswith('Util')
                or field == 'percentBcGt75'
                or field == 'bcOpenToBuy'
                or field == 'empLength'):
            return NEGATIVE_INT 
        elif field=='empTitle':
            return ''
        else:
            return None
    
    def null_fill_fields(self):
        return [f for f in self.api_fields if self.null_fill_value(f) is not None]

    def parse(self, data):
        for k in self.api_fields:
            if k not in data.keys():
                print 'Field {} is missing'.format(k)
       
        #API empLength is given in months. Convert to years
        if data['empLength'] not in range(-1, 11):
            data['empLength'] = min(11, data['empLength'] / 12)
 
        for k,v in data.items():
            if v is None:
                data[k] = self.null_fill_value(k) 
                
            if type(v) in [str, unicode]:
                if 'String' not in k:
                    data[k] = self.string_converter.convert(k, v)                
                    if data[k] != v:
                        data[u'{}String'.format(k)] = v

            if data[k] is None and k not in self.ok_to_be_null:
                print 'Field {} has a null value'.format(k)



class LocationDataManager(object):
    def __init__(self):

        self.bls = load_bls()
        self.census = load_census_data()
        self.z2f = defaultdict(lambda :list(), load_z2f())
        self.z2c = load_z2c()
        self.z2pc = load_z2primarycity()
        self.metro = load_metro_housing()
        self.nonmetro = load_nonmetro_housing()

    def get_zip_features(self, zip3):
        ''' takes the first three digits of the zip code and returns
        a dictionary of features for that location'''
        info = dict()
        ur, avg_ur, ur_chg, ur_range = get_urate(self.bls, zip3)
        info['urate'] = ur
        info['avg_urate'] = avg_ur
        info['urate_chg'] = ur_chg
        info['urate_range'] = ur_range
        info['med_income'] = get_income_data(self.census, self.z2f[zip3])
        info['primaryCity'] = self.z2pc[zip3]
        metro_hpa = self.metro.ix[self.z2c[zip3]].dropna()
        if len(metro_hpa)>0:
            info['HPA1Yr'] = metro_hpa['1yr'].mean()
            info['HPA5Yr'] = metro_hpa['5yr'].mean()
        else:
            print 'No FHFA data found for zip code {}xx'.format(zip3)
            info['HPA1Yr'] = 0
            info['HPA5Yr'] = 0
        #    nonmetro_hpa = self.nonmetro.ix[[loan['state']]]
        #    info['HPA1Yr'] = nonmetro_hpa['1yr'].values[0]
        #    info['HPA5Yr'] = nonmetro_hpa['5yr'].values[0]
        return info


class LogOddsCalculator(object):
    def __init__(self, log_odds_dict, tok_type):
        self.log_odds_dict = defaultdict(lambda :0, log_odds_dict)
        self.tok_type = tok_type

    def calc_log_odds(self, x):
        if len(x)==0:
            return 0
        if self.tok_type=='word':
            x = x.replace('^','').replace('$','')
            toks = x.split()
        else: 
            tok_len = self.tok_type
            toks = np.unique([x[i:i+tok_len] for i in range(max(1,len(x)-tok_len+1))])
        log_odds = np.sum(map(lambda x:self.log_odds_dict[x], toks))
        return log_odds
 


def construct_loan_dict(grade, term, rate, amount):
    pmt = np.pmt(rate/1200., term, amount)
    loan = dict([('grade', grade),('term', term),('monthly_payment', abs(pmt)),
        ('loan_amount', amount), ('intRate', rate)])
    return loan



#TODO: modify this to use the rev_util version of prepayment curves, and adjust the curves
# by hinge tilting them from month 20 & 30 for 3-year and 5-year loans, resp.
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
        plt.figure()
        plt.plot(empirical_default, 'b')
        plt.plot(smoothed_default, 'r')
        plt.title(i)
        plt.grid()
        plt.show()

    all_grades = list('ABCDEFG')
    for grade in all_grades:
        plt.figure()
        plt.plot(default_curves['{}60'.format(grade)], 'b')
        plt.plot(default_curves['{}36'.format(grade)], 'r')
        plt.title(grade)
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
    z2f = json.load(file(os.path.join(reference_dir, 'zip3_fips.json'),'r'))

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
        
