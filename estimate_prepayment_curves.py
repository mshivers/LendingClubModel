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
    last = g_id.last()
    print 'After first()', (now() - t).total_seconds()
    loan_amt = first['PBAL_BEG_PERIOD']
    loan_amt.name = 'loan_amt'

    if 'loan_amt' not in df.columns:
        df = df.join(loan_amt, on='LOAN_ID')
        df['prepay_pct'] = df['prepay_amt'] / df['loan_amt']
    df['issue_year'] = df['IssuedDate'].apply(lambda x: int(x[-4:]))
    prepays = df.pivot(index='LOAN_ID', columns='MOB', values='prepay_pct') 

    # combine all payments for MOB=0 (a very small number of prepayments before the first payment is due) with MOB=1
    prepays[1] = prepays[1] + prepays[0].fillna(0)
    del prepays[0]

    prepays = prepays.join(last[['term', 'grade', 'IssuedDate', 'MOB']])
    #combine E, F & G (there's not many of them
    prepays['grade'] = prepays['grade'].apply(lambda x: min(x, 'E'))

    prepays = prepays.sort('IssuedDate')
    for d in set(prepays['IssuedDate'].values):
        idx = prepays.IssuedDate == d
        max_mob = prepays.ix[idx, 'MOB'].max()
        prepays.ix[idx, :(max_mob-1)] = prepays.ix[idx, :(max_mob-1)].fillna(0)
        print d, max_mob
        print prepays.ix[idx, :max_mob].sum(0)

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

prepay_curves = dict()
for i, r in mean_prepays.iterrows():
    N = i[0]
    win = 19 if N==36 else 29 
    empirical_prepays = r[:N+1].values
    smoothed_prepays = scipy.signal.savgol_filter(empirical_prepays, win , 3)
    smoothed_prepays = np.maximum(0, smoothed_prepays) 
    prepay_curves['{}{}'.format(i[1], i[0])] = list(np.cumsum(smoothed_prepays))
    plt.figure()
    plt.plot(empirical_prepays, 'b')
    plt.plot(smoothed_prepays, 'r')
    plt.title(i)
    plt.grid()
    plt.show()


for grade in list('ABCDE'):
    plt.figure()
    plt.plot(prepay_curves['{}60'.format(grade)], 'b')
    plt.plot(prepay_curves['{}36'.format(grade)], 'r')
    plt.title(grade)
    plt.grid()
    plt.show()

for term in [36,60]:
    plt.figure()
    for grade in list('ABCDE'):
        plt.plot(prepay_curves['{}{}'.format(grade,term)])
    plt.title('Term: {}'.format(term))
    plt.grid()
    plt.show()

json.dump(prepay_curves, open('/Users/marcshivers/LCModel/prepay_curves.json', 'w'), indent=4)


