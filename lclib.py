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


LARGE_INT = 9999 
NEGATIVE_ONE = -1 

#parent_dir = '/home/apprun/LCModel/LCModel'
parent_dir = '/Users/marcshivers/LCModel'
data_dir = os.path.join(parent_dir, 'data')

update_hrs = [1,5,9,13,17,21]
oos_cutoff = np.datetime64('2015-01-01')

# clean dataframe
home_map = dict([('ANY', 0), ('NONE',0), ('OTHER',0), ('RENT',1), ('MORTGAGE',2), ('OWN',3)])

purpose_dict = defaultdict(lambda :np.nan)
purpose_dict.update([('credit_card', 0), ('credit_card_refinancing', 0), 
                    ('debt_consolidation',1), ('home_improvement',2), 
                    ('car',3), ('car_financing',3), 
                    ('educational',4), 
                    ('house',5), ('home_buying',5),
                    ('major_purchase',6), 
                    ('medical_expenses',7), ('medical',7), 
                    ('moving',8), ('moving_and_relocation',8), 
                    ('other',9),
                    ('renewable_energy',10), ('green_loan',10),
                    ('business',11),('small_business',11),
                    ('vacation',12), ('wedding',13)])

def purpose_mapping(prp):
    prp = prp.lower().replace(' ', '_')
    if prp in purpose_dict.keys():
        p_num = purpose_dict[prp]
    else:
        p_num = np.nan
        print 'Loan purpose = {} not found in purpose_map keys.\n'.format(prp)
    return p_num

subgrades = ['{}{}'.format(l,n) for l in 'ABCDEFG' for n in range(1,6)]
subgrade_map = defaultdict(lambda :np.nan, zip(subgrades, range(len(subgrades))))

# Must be set to 1 iff True; LC api only give true/false
verification_dict = dict([('Verified',2), ('Source Verified',1), ('Not Verified',0)]) 
api_verification_dict = dict([('VERIFIED',2), ('SOURCE_VERIFIED',1), ('NOT_VERIFIED',0)])

init_status_dict = dict([('f',0), ('w',1)])


def substrings(x):
    toks = list()
    x = '^{}$'.format(x)
    #add all whole words
    toks.extend(x.split())
    toks.extend([x[i:i+k] for k in range(1,len(x)+1) 
        for i in range(max(1,len(x)-k+1))])
    return list(set(toks))

def employment_length_map(el):
    el=el.replace('< 1 year', '0')
    el=el.replace('1 year','1')
    el=el.replace('10+ years', '11')
    el=el.replace('n/a', '-1')
    el=el.replace(' years', '')
    return int(el)

def clean_title(x):
    x = str(x).strip().lower()
    if x=='n/a':
        x = 'blank'
    x = x.replace("'","")
    x = x.replace('"','')
    x = x.replace('/', ' ')
    for tok in '`~!@#$%^&*()_-+=\|]}[{;:/?.>,<':
        x = x.replace(tok,'')
    return x

def load_training_data(full_dataset=False):
    fname = os.path.join(data_dir, 'LoanStats3{}_securev1.csv')
    da = pd.read_csv(fname.format('a'), header=1, nrows=39786)
    db = pd.read_csv(fname.format('b'), header=1)
    dc = pd.read_csv(fname.format('c'), header=1)
    dd = pd.read_csv(fname.format('d'), header=1)
    df = pd.concat((da,db,dc,dd), ignore_index=True)
    idx = (df[['last_pymnt_d', 'issue_d']].isnull()).any(1)
    df = df.ix[~idx]

    # only keep titles in ascii
    idx = df['emp_title'].apply(lambda x:max(ord(c) for c in str(x))<128)
    df = df.ix[idx]
    
    df['id'] = df['id'].astype(int)

    # clean dataframe
    cvt = dict()
    cvt['term'] = lambda x: int(x.strip().split(' ')[0])
    cvt['int_rate'] = lambda x: float(x[:-1])
    cvt['zip_code'] = lambda x: float(x[:3])
    cvt['revol_util'] = lambda x: np.nan if '%' not in str(x) else min(100,round(float(x[:-1]),0))
    cvt['revol_bal'] = lambda x: round(x,-2)
    cvt['dti'] = lambda x: round(x,0)
    cvt['grade'] = lambda x: 'ABCDEFG'.index(str(x))
    cvt['issue_d'] = lambda x: dt.strptime(x, '%b-%Y')
    cvt['last_pymnt_d'] = lambda x: dt.strptime(x, '%b-%Y')
    cvt['earliest_cr_line'] = lambda x: dt.strptime(x, '%b-%Y')
    cvt['home_ownership'] = lambda x: home_map[x]
    cvt['purpose'] = lambda x: purpose_mapping(x)
    cvt['sub_grade'] = lambda x: subgrade_map[x]
    cvt['emp_length'] = lambda x: employment_length_map(x)
    cvt['verification_status'] = lambda x: verification_dict[x]
    cvt['desc'] = lambda x: float(len(str(x)) > 3)
    cvt['initial_list_status'] = lambda x: init_status_dict[x] 

    for col in [c for c in df.columns if c in cvt.keys()]:
        df[col] = df[col].apply(cvt[col])
   
    df['mths_since_last_record'] = df['mths_since_last_record'].fillna(LARGE_INT)
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(LARGE_INT)
    df['mths_since_last_major_derog'] = df['mths_since_last_major_derog'].fillna(LARGE_INT)

    df['wgt_default'] = 0.0 
    df.ix[df['loan_status']=='In Grace Period', 'wgt_default'] = 0.28
    df.ix[df['loan_status']=='Late (16-30 days)', 'wgt_default'] = 0.58
    df.ix[df['loan_status']=='Late (31-120 days)', 'wgt_default'] = 0.74
    df.ix[df['loan_status']=='Default', 'wgt_default'] = 0.89
    df.ix[df['loan_status']=='Charged Off', 'wgt_default'] = 1.0

    one_year = 360*24*60*60*1e9  # we want to find payments strictly less than 1 year, so we use 360 days here.
    df['12m_late'] = (df['wgt_default']>0) & (df['last_pymnt_d']-df['issue_d']<one_year)
    df['12m_wgt_default'] = df['12m_late'] * df['wgt_default']

    df['emp_title'] = df['emp_title'].fillna('Blank').apply(lambda x:str(x).strip())
    df['clean_title'] = df['emp_title'].apply(lambda x:clean_title(x))

    df['mths_since_last_delinq'].fillna(LARGE_INT)
    df['mths_since_last_major_derog'].fillna(LARGE_INT)
    df['mths_since_last_record'].fillna(LARGE_INT)

    # use only data after emp_title switched from company name
    if not full_dataset:
        endx = df['issue_d']>=np.datetime64('2013-10-01')
        df = df.ix[endx].copy()

    return df

def only_ascii(s):
    return ''.join([c for c in s if ord(c)<128])

def calc_regularized_log_odds(x, log_odds_dict, tok_len):
    if len(x)==0:
        return 0
    toks = [x[i:i+tok_len] for i in range(max(1,len(x)-tok_len+1))]
    tok_odds = np.array(map(lambda x:log_odds_dict[x], toks))
    tok_odds[1+np.where(tok_odds[:-1]==0)[0]] = 0
    tok_odds[np.where(tok_odds[1:]==0)[0]] = 0
    return np.mean(tok_odds[tok_odds!=0])


def calc_log_odds(x, log_odds_dict, tok_len):
    if len(x)==0:
        return 0
    toks = np.unique([x[i:i+tok_len] for i in range(max(1,len(x)-tok_len+1))])
    log_odds = np.sum(map(lambda x:log_odds_dict[x], toks))
    return log_odds

def load_tok4_clean_title_log_odds_func():
    ''' loads a dictionary mapping each 4-letter string into a log-odds value. '''
    data = json.load(open(os.path.join(parent_dir,'prod_tok4_clean_title_log_odds.json'),'r'))
    data = defaultdict(lambda :0, data)
    def calc_log_odds(x, log_odds_dict, tok_len):
        if len(x)==0:
            return 0
        toks = [x[i:i+tok_len] for i in range(max(1,len(x)-tok_len+1))]
        log_odds = np.mean(map(lambda x:log_odds_dict[x], toks))
        return log_odds
    def log_odds_func(x):
        return calc_log_odds(x, data, tok_len=4)
    return log_odds_func

def load_tok4_emp_title_log_odds_func():
    ''' loads a dictionary mapping each 4-letter string into a log-odds value. '''
    data = json.load(open(os.path.join(parent_dir,'tok4_recent_employer_title_log_odds.json'),'r'))
    data = defaultdict(lambda :0, data)
    def calc_log_odds(x, log_odds_dict, tok_len):
        if len(x)==0:
            return 0
        toks = [x[i:i+tok_len] for i in range(max(1,len(x)-tok_len+1))]
        log_odds = np.mean(map(lambda x:log_odds_dict[x], toks))
        return log_odds
    def log_odds_func(x):
        return calc_log_odds(x, data, tok_len=4)
    return log_odds_func

def load_tok4_emp_name_log_odds_func():
    ''' loads a dictionary mapping each 4-letter string into a log-odds value. '''
    data = json.load(open(os.path.join(parent_dir,'tok4_employer_name_log_odds.json'),'r'))
    data = defaultdict(lambda :0, data)
    def log_odds_func(x):
        return calc_log_odds(x, data, tok_len=4)
    return log_odds_func

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


def load_tok4_capitalization_log_odds_func():
    ''' loads a dictionary mapping each 4-letter string into a log-odds value. '''
    data = json.load(open(os.path.join(parent_dir,'new_capitalization_log_odds.json'),'r'))
    data = defaultdict(lambda :0, data)
    def log_odds_func(x):
        cap = tokenize_capitalization(x)
        return calc_log_odds(cap, data, tok_len=4)
    return log_odds_func


def create_clean_title_log_odds_json(df, fname='ctlo.json', tok_len=4, fld_name='clean_title'):
    #employer name was in the emp_title field before 9/24/13
    
    idx = (df['issue_d']>=np.datetime64('2013-10-01')) & (df['issue_d']<oos_cutoff)
    training = df[idx].copy()

    toks = list()
    titles = training[fld_name].apply(lambda x:'^{}$'.format(x)) #add string boundary tokens

    for ct in titles.values:
        toks.extend([ct[i:i+tok_len] for i in range(max(1,len(ct)-tok_len+1))])

    tok_df = pd.DataFrame(Counter(toks).items(), columns=['tok', 'freq'])
    tok_df = tok_df.sort('freq',ascending=False)
    
    odds_map = dict()
    mean_default = training['12m_wgt_default'].mean() 
    C = 2000 #regularized number of mean defaults
    for _, row in tok_df.iterrows():
        tok, freq = row
        if freq < 1000:
            continue 
        training['has_tok'] = titles.apply(lambda x: tok in x)
        grp = training.groupby('has_tok')
        default_sum = grp.sum()['12m_wgt_default'] + C * mean_default
        default_count = grp.count()['12m_wgt_default'] + C
        regularized_default = default_sum  / default_count 
        log_odds = np.log(regularized_default[True]) - np.log(regularized_default[False])
        print default_count[True], '"{}"'.format(tok), '{:1.2f}'.format(log_odds)
        print regularized_default
        print '\n'
        odds_map[tok] = log_odds

        json.dump(odds_map, open(os.path.join(parent_dir, fname),'w'))
    return odds_map


def create_capitalization_log_odds_json(df, fname='caplo.json', tok_len=4, fld_name='emp_title'):
    idx = (df['issue_d'] < oos_cutoff)
    df = df[idx].copy()

    toks = list()
    df['tokenized'] = df[fld_name].apply(tokenize_capitalization)
    for ct in df['tokenized'].values:
       toks.extend([ct[i:i+tok_len] for i in range(max(1,len(ct)-tok_len+1))])

    tok_df = pd.DataFrame(Counter(toks).items(), columns=['tok', 'freq'])
    tok_df = tok_df.sort('freq',ascending=False)

    odds_map = dict()
    mean_default = df['12m_wgt_default'].mean() 
    C = 2000 #regularized number of mean defaults
    for _, row in tok_df.iterrows():
        tok, freq = row
        if freq < 5000:
            continue 
        df['has_tok'] = df['tokenized'].apply(lambda x: tok in x)
        grp = df.groupby('has_tok')
        default_sum = grp.sum()['12m_wgt_default'] + C * mean_default
        default_count = grp.count()['12m_wgt_default'] + C
        regularized_default = default_sum  / default_count 
        log_odds = np.log(regularized_default[True]) - np.log(mean_default)
        print default_count[True], '"{}"'.format(tok), '{:1.2f}'.format(log_odds)
        print regularized_default
        print '\n'
        odds_map[tok] = log_odds

        json.dump(odds_map, open(os.path.join(parent_dir, fname),'w'))
    return odds_map


def create_emp_name_log_odds_json(df, fname, tok_len=4, fld_name='emp_title'):
    #employer name was in the emp_title field before 9/24/13
    training = df[df['issue_d'] < np.datetime64('2013-10-01')].copy()
    toks = list()
    for ct in training[fld_name].values:
        toks.extend([ct[i:i+tok_len] for i in range(max(1,len(ct)-tok_len+1))])

    tok_df = pd.DataFrame(Counter(toks).items(), columns=['tok', 'freq'])
    tok_df = tok_df.sort('freq',ascending=False)

    odds_map = dict()
    mean_default = training['12m_wgt_default'].mean() 
    C = 1000 #regularized number of mean defaults
    for _, row in tok_df.iterrows():
        tok, freq = row
        if freq < 200:
            continue 
        training['has_tok'] = training[fld_name].apply(lambda x: tok in x)
        grp = training.groupby('has_tok')
        default_sum = grp.sum()['12m_wgt_default'] + C * mean_default
        default_count = grp.count()['12m_wgt_default'] + C
        regularized_default = default_sum  / default_count 
        log_odds = np.log(regularized_default[True]) - np.log(regularized_default[False])
        print default_count[True], '"{}"'.format(tok), '{:1.2f}'.format(log_odds)
        print regularized_default
        print '\n'
        odds_map[tok] = log_odds

        json.dump(odds_map, open(os.path.join(parent_dir, fname),'w'))
    return odds_map



def create_clean_title_dataset(df, fname, tok_len=4, fld_name='clean_title'):
    #employer name was in the emp_title field before 9/24/13
    
    idx = (df['issue_d']>=np.datetime64('2013-10-01')) & (df['issue_d']<oos_cutoff)
    training = df[idx].copy()

    toks = list()
    titles = training[fld_name].apply(lambda x:'^{}$'.format(x)) #add string boundary tokens

    for ct in titles.values:
        toks.extend([ct[i:i+tok_len] for i in range(max(1,len(ct)-tok_len+1))])

    tok_df = pd.DataFrame(Counter(toks).items(), columns=['tok', 'freq'])
    tok_df = tok_df.sort('freq',ascending=False)
    
    odds_map = dict()
    mean_default = training['12m_wgt_default'].mean() 
    for _, row in tok_df.iterrows():
        tok, freq = row
        training['has_tok'] = titles.apply(lambda x: tok in x)
        grp = training.groupby('has_tok')
        default_sum = grp.sum()['12m_wgt_default']
        default_count = grp.count()['12m_wgt_default']
        default = default_sum  / default_count 
        log_odds = np.log(default[True]) - np.log(default[False])
        print default_count[True], '"{}"'.format(tok), '{:1.2f}'.format(log_odds)
        print default
        print '\n'
        odds_map[tok] = (default_sum[True], default_count[True], default_count[False]) 
    df2 = pd.DataFrame(odds_map).T
    df2.columns = ['default_wgt_sum', 'default_count', 'current_count']
    df2.to_csv(open(os.path.join(parent_dir, fname),'w'))
    return df2 



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

def prestage(loan, min_rate, min_income):
    loanRate = float(loan['loanRate'])
    income = float(loan['grossIncome'].split('/')[0].replace('$','').replace(',',''))
    return (loanRate>=min_rate) and (income>min_income) and (loan['inquiriesLast6Months']==0)


def invest_amount(irr, min_irr, max_invest=None):
    if max_invest==None:
        max_invest = 500
    if irr < min_irr:
        return 0
    else:
        return min(max_invest, 100 * np.ceil(1*(irr - min_irr)))

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
    l = loan 

    pstr = 'IRR: {:1.2f}%'.format(100*l['irr'])
    pstr += ' | Alpha: {:1.2f}%'.format(l['alpha'])
    pstr += ' | IntRate: {}%'.format(l['int_rate'])
    pstr += ' | Staged: ${}'.format(l['staged_amount'])

    pstr += '\nDefaultRisk: {:1.2f}%'.format(l['default_risk'])
    pstr += ' | DefaultMax: {:1.2f}%'.format(l['default_max'])
    pstr += ' | DefaultStd: {:1.2f}%'.format(l['default_std'])
    pstr += ' | RiskFactor: {:1.2f}'.format(l['risk_factor'])

    pstr += '\nLoanAmnt: ${:1,.0f}'.format(l['loanAmount'])
    pstr += ' | Term: {}'.format(l['term'])
    pstr += ' | Grade: {}'.format(l['subGrade'])
    pstr += ' | Purpose: {}'.format(l['purpose'])
    pstr += ' | LoanId: {}'.format(l['id'])
    
    pstr += '\nRevBal: ${:1,.0f}'.format(l['revolBal'])
    pstr += ' | RevUtil: {}%'.format(l['revolUtil'])
    pstr += ' | DTI: {}%'.format(l['dti'])
    pstr += ' | Inq6m: {}'.format(l['inqLast6Mths'])
    pstr += ' | 1stCredit: {}'.format(l['earliestCrLine'].split('T')[0])
    pstr += ' | fico: {}'.format(l['ficoRangeLow'])
  
    pstr += '\nJobTitle: {}'.format(l['currentJobTitle'])
    pstr += ' | Company: {}'.format(l['currentCompany'])
    
    pstr += '\nClean Title Log Odds: {:1.2f}'.format(l['clean_title_log_odds'])
    pstr += ' | Capitalization Log Odds: {:1.2f}'.format(l['capitalization_log_odds'])
    pstr += ' | Income: ${:1,.0f}'.format(l['annualInc'])
    pstr += ' | Tenure: {}'.format(l['empLength'])

    pstr += '\nLoc: {},{}'.format(l['addrZip'], l['addrState'])
    pstr += ' | MedInc: ${:1,.0f}'.format(l['med_income'])
    pstr += ' | URate: {:1.1f}%'.format(100*l['urate'])
    pstr += ' | 12mChg: {:1.1f}%'.format(100*l['urate_chg'])
    pstr += ' | 12mRange: {:1.1f}%'.format(100*l['urate_range'])

    pstr += '\nHomeOwn: {}'.format(l['homeOwnership'])
    pstr += ' | PrimaryCity: {}'.format(l['primaryCity'])
    pstr += ' | HPA1: {:1.1f}%'.format(l['HPA1Yr'])
    pstr += ' | HPA5: {:1.1f}%'.format(l['HPA5Yr'])

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
    l = loan
    all_details = '\n'.join(sorted(['{}: {}'.format(k,v) for k,v in l.items() 
        if k in email_keys or k.startswith('dflt')]))
    return all_details 


def send_email(msg):
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login('marc.shivers@gmail.com', 'udxqmiraraiudjux')
    msg_list = ['From: marc.shivers@gmail.com', 
                       'To: marc.shivers@gmail.com', 
                       'Subject: Lending Club', '']
    msg_list.append(msg)
    msg = '\r\n'.join(msg_list) 
    server.sendmail('marc.shivers@gmail.com',['marc.shivers@gmail.com'],msg)
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
        result = {'irr_count': x['irr'].count(), 'irr_mean': x['irr'].mean()}
        return pd.Series(result, name='metrics')

    msg_list.append(g.apply(compute_metrics).to_string())
    msg_list.append(g.count().to_string())
    msg_list.append(g.mean().to_string())
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
    fname = os.path.join(data_dir, 'lya2014.txt')
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
        data.to_csv(os.path.join(data_dir, 'HPI_AT_nonmetro.csv'))
    except:
        data = pd.read_csv(os.path.join(data_dir,'HPI_AT_nonmetro.csv'))
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
        data.to_csv(os.path.join(data_dir, 'HPI_AT_metro.csv'))
    except:
        data = pd.read_csv(os.path.join(data_dir,'HPI_AT_metro.csv'), skiprows=1, header=None, names=cols)
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

    bls_fname = os.path.join(data_dir, 'bls_summary.csv')
    if os.path.exists(bls_fname):
        update_dt = dt.fromtimestamp(os.path.getmtime(bls_fname))
        days_old = (dt.now() - update_dt).days 
        summary = pd.read_csv(bls_fname)
    else:
        days_old = 999

    if days_old > 14:
        try:
            link = 'http://www.bls.gov/lau/laucntycur14.txt'
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
    data = pd.read_csv(os.path.join(data_dir, 'z2c.csv'))
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
    data = json.load(open(os.path.join(parent_dir, 'zip2location.json'), 'r'))
    return dict([int(k), locs] for k, locs in data.items())


def load_z2primarycity():
    ''' loads a dictionary mapping the first 3 digits of the zip code to the
    most common city with that zip code prefix'''
    data = json.load(open(os.path.join(parent_dir,'zip2primarycity.json')))
    return dict([int(k), locs] for k, locs in data.items())


def load_z2f():
    ''' loads a dictionary mapping the first 3 digits of the zip code to a 
    list of FIPS codes'''
    d = json.load(open(os.path.join(parent_dir, 'z2f.json'), 'r'))
    return dict([int(k), [int(v) for v in vals]] for k, vals in d.items())


def load_f2c():
    ''' loads a dictionary mapping FIPS codes to CBSA codes and names '''
    #need a mapping that lists NY-NJ to 35614, rather than 36544
    d = json.load(open(os.path.join(parent_dir, 'fips2cbsa.json'), 'r'))
    return dict([int(k), [int(v[0]), v[1]]] for k, v in d.items())


def save_loan_info(loans):
    f = open(os.path.join(data_dir, 'employer_data.csv'),'a')
    for l in loans:
        f.write('{}|{}|{}\n'.format(l['id'], l['currentJobTitle'],l['currentCompany']))
        l['details_saved']=True
    f.close()

    f = open(os.path.join(data_dir, 'all_api_data.csv'),'a')
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


bls = load_bls()
census = load_census_data()
z2f = defaultdict(lambda :list(), load_z2f())
z2c = load_z2c()
z2pc = load_z2primarycity()
metro = load_metro_housing()
nonmetro = load_nonmetro_housing()

def add_external_features(l):
    ''' Add all the external data sources to the loan details'''

    zip3 = int(l['zip3'])
    ur, avg_ur, ur_chg, ur_range = get_urate(bls, zip3)
    l['urate'] = ur
    l['avg_urate'] = avg_ur
    l['urate_chg'] = ur_chg
    l['urate_range'] = ur_range
    l['med_income'] = get_income_data(census, z2f[zip3])
    l['primaryCity'] = z2pc[zip3]
    metro_hpa = metro.ix[z2c[zip3]].dropna()
    if len(metro_hpa)>0:
        l['HPA1Yr'] = metro_hpa['1yr'].mean()
        l['HPA5Yr'] = metro_hpa['5yr'].mean()
    else:
        nonmetro_hpa = nonmetro.ix[[l['state']]]
        l['HPA1Yr'] = nonmetro_hpa['1yr'].values[0]
        l['HPA5Yr'] = nonmetro_hpa['5yr'].values[0]


default_curves = json.load(open(os.path.join(parent_dir, 'default_curves.json'), 'r'))
prepay_curves = json.load(open(os.path.join(parent_dir, 'prepay_curves.json'), 'r'))


def construct_loan_dict(grade, term, rate, amount):
    pmt = np.pmt(rate/1200., term, amount)
    return dict([('grade', grade),('term', term),('monthly_payment', abs(pmt)),
        ('loan_amount', amount), ('int_rate', rate)])




def calc_npv(l, discount_rate=0.10):
    ''' All calculations assume a loan amount of $1.
    Note the default curves are the cumulative percent of loans that have defaulted prior 
    to month m; the prepayment curves are the average percentage of outstanding balance that is prepaid 
    each month (not cumulative) where the average is over all loans that have not yet matured (regardless 
    of prepayment or default).  We'll assume that the prepayments are in full'''

    net_payment_pct = 0.99  #LC charges 1% fee on all incoming payments

    key = '{}{}'.format(min('G', l['grade']), l['term']) 
    prepay_rate = np.array(prepay_curves[key])
    base_defaults = np.array(default_curves[key])

    #adjusted_default_risk = (l['default_risk'] + l['default_std'])/100
    adjusted_default_risk = l['default_max']/100 
    risk_factor = adjusted_default_risk / base_defaults[11]
    l['risk_factor'] = risk_factor

    # adjust only the first 15 months of default rates downward to match the model's 12-month default estimate.
    # this is a hack; adjusting the entire curve seems obviously wrong.  E.g if we had a C default curve
    # that was graded D, adjusting the entire D curve down based on the 12-mth ratio would underestimate defaults
    cdefaults = np.r_[base_defaults[:1],np.diff(base_defaults)]
    if risk_factor < 1:
        cdefaults[:15] *= risk_factor
    else:
        cdefaults *= risk_factor

    cdefaults = cdefaults.cumsum()
    
    monthly_int_rate = l['int_rate']/1200.
    monthly_discount_rate = (1 + discount_rate) ** (1/12.) - 1
    monthly_payment = l['monthly_payment'] / l['loan_amount']

    # start with placeholder for time=0 investment for irr calc later
    payments = np.zeros(l['term']+1)
    
    principal_balance = 1
    # add monthly payments
    for m in range(1, l['term']+1):

        interest_due = principal_balance * monthly_int_rate
        principal_due = monthly_payment - interest_due

        # prepayment rate is a pct of ending balance
        principal_balance -= principal_due
        prepayment_amt = principal_balance * prepay_rate[m-1]
        
        scheduled_amt = monthly_payment * (1 - cdefaults[m-1])
        payments[m] = prepayment_amt + scheduled_amt
        
        # reduce monthly payment to reflect this month's prepayment
        principal_balance -= prepayment_amt
        monthly_payment *= (1 - prepay_rate[m-1])


    # reduce payments by lending club service charge
    payments *= net_payment_pct
    npv = np.npv(monthly_discount_rate, payments) 

    # Add initial investment outflow at time=0 to calculate irr: 
    payments[0] += -1
    irr = np.irr(payments)
    
    l['irr'] = -1 + (1 + irr) ** 12
    l['npv'] = 100 * npv    

    return 
    


    

class LC():
    def __init__(self):
        self.zip2feature = dict()

