import pandas as pd
import numpy as np
from datetime import datetime  as dt
from collections import Counter, defaultdict
import scipy.signal
from matplotlib import pyplot as plt
import json

now = dt.now
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
t = now()

if 'df' not in locals().keys():
    cols = ['LOAN_ID', 'PBAL_BEG_PERIOD', 'PBAL_END_PERIOD', 'MONTHLYCONTRACTAMT', 'InterestRate', 
            'VINTAGE', 'IssuedDate', 'RECEIVED_AMT', 'DUE_AMT', 'PERIOD_END_LSTAT', 'MOB', 'term', 'grade']
    df = pd.read_csv('/Users/marcshivers/LCModel/data/PMTHIST_ALL_20160117_v1.csv', 
            sep=',') #, usecols=cols)
    print 'After read_csv',  (now() - t).total_seconds()
    df['prepay_amt'] = np.maximum(0, df['RECEIVED_AMT'] - df['DUE_AMT'])

    g_id = df.groupby('LOAN_ID')
    print 'After groupby by ID', (now() - t).total_seconds()

    first = g_id.first()
    print 'After first()', (now() - t).total_seconds()
    loan_amt = first['PBAL_BEG_PERIOD']
    loan_amt.name = 'loan_amt'

    if 'loan_amt' not in df.columns:
        df = df.join(loan_amt, on='LOAN_ID')
        df['prepay_pct'] = df['prepay_amt'] / df['loan_amt']
df['issue_year'] = df['IssuedDate'].apply(lambda x: int(x[-4:]))
prepays = df.ix[df.issue_year>2012].pivot(index='LOAN_ID', columns='MOB', values='prepay_pct') 
# combine all payments for MOB=0 (a very small number of prepayments before the first payment is due) with MOB=1
prepays[1] = prepays[1] + prepays[0].fillna(0)
del prepays[0]

prepays = prepays.join(first[['term', 'grade']])
#combine E, F & G (there's not many of them
prepays['grade'] = prepays['grade'].apply(lambda x: min(x, 'E'))

mean_prepays = prepays.groupby(['term', 'grade']).mean()

for N in [36, 60]:
    plt.figure()
    for i, r in mean_prepays.iterrows():
        if i[0]==N:
            plt.plot(r.cumsum()[:N])
    plt.legend(list('ABCDEFG'), loc=2)
    plt.title(N)
    plt.grid()
    plt.show()

print 'After prepay pivot', (now() - t).total_seconds()
for N in [36, 60]:
    plt.figure()
    for i, r in mean_prepays.iterrows():
        if N == i[0]:
            win = 19 if N==36 else 29 
            empirical_prepays = r[:N].values
            smoothed_prepays = scipy.signal.savgol_filter(empirical_prepays, win , 3)
            smoothed_prepays = np.maximum(0, smoothed_prepays) 
            #plt.plot(empirical_prepays, 'b')
            plt.plot(smoothed_prepays)
    plt.title(N)
    plt.legend(list('ABCDEFG'), loc=1)
    plt.grid()
    plt.show()

1/0
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
       defaults[i, 2+months_paid] = num
    
cols = ['num_loans', 'max_age'] + range(61)
defaults = pd.DataFrame(data=defaults, index=index, columns=cols).reset_index()   
issued = pd.DataFrame(data=issued, index=index, columns=cols).reset_index()    

defaults['grade'] = np.minimum(defaults['grade'], 'E')
issued['grade'] = np.minimum(issued['grade'], 'E')

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


for grade in list('ABCDE'):
    plt.figure()
    plt.plot(default_curves['{}60'.format(grade)], 'b')
    plt.plot(default_curves['{}36'.format(grade)], 'r')
    plt.title(grade)
    plt.grid()
    plt.show()

for term in [36,60]:
    plt.figure()
    for grade in list('ABCDE'):
        plt.plot(default_curves['{}{}'.format(grade,term)])
    plt.title('Term: {}'.format(term))
    plt.grid()
    plt.show()

json.dump(default_curves, open('/Users/marcshivers/LCModel/default_curves.json', 'w'), indent=4)


