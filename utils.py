from constants import UPDATE_HOURS
from datetime import datetime as dt
from datetime import timedelta as td 
import numpy as np

def only_ascii(s):
    return ''.join([c for c in s if ord(c)<128])

def reset_time():
    try:
        import os
        print 'Attempting to adjust system time'
        response = os.popen('sudo ntpdate -u time.apple.com')
        print response.read()
        print 'Reset Time Successfully!'  
    except:
        print 'Failed to reset system time' 


def clean_title(x):
    x = str(x).strip().lower()
    x = x.replace("'","")
    x = x.replace('"','')
    x = x.replace('/', ' ')
    for tok in '`~!@#$%^&*()_-+=\|]}[{;:/?.>,<':
        x = x.replace(tok,'')
    x = '^{}$'.format(x) 
    return x


def format_title(x):
    return '^{}$'.format(only_ascii(x))

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

def hourfrac(tm):
    return (tm.hour + tm.minute/60.0 + tm.second / 3600.0)


def sleep_seconds(win_len=30):
     # win_len is the number of seconds to continuously check for new loans.
     # The period ends at the official update time.  
     now = dt.now()
     tm = now + td(seconds=win_len/2.0)
     update_seconds = np.array([60*60*(hr - hourfrac(tm)) for hr in UPDATE_HOURS])
     center = min(abs(update_seconds))
     max_sleep_seconds = 0.8 * max(center - win_len, 0)
     #seconds_to_hour = (59 - now.minute) * 60.0 + (60 - now.second)
     return max_sleep_seconds


def save_charity_pct():
    irs = pd.read_csv('/Users/marcshivers/Downloads/12zpallagi.csv')
    irs['zip3'] = irs['zipcode'].apply(lambda x:int(x/100))
    irs = irs.ix[irs['AGI_STUB']<5]
    grp = irs.groupby('zip3')
    grp_sum = grp.sum()
    tax_df = pd.DataFrame({'agi':grp_sum['A00100'], 'charity':grp_sum['A19700']})
    tax_df['pct'] = tax_df['charity'] * 1.0 / tax_df['agi']
    json.dump(tax_df['pct'].to_dict(), open(os.path.join(reference_data_dir, 'charity_pct.json'), 'w'))


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


