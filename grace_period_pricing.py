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

    g_id = df.groupby('LOAN_ID')
    print 'After groupby by ID', (now() - t).total_seconds()


    1/0
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

    join_cols = ['term', 'grade', 'IssuedDate', 'MOB', 'dti', 'HomeOwnership', 'MonthlyIncome', 'EmploymentLength']
    prepays = prepays.join(last[join_cols])

    loan_amount = first['PBAL_BEG_PERIOD']
    loan_amount.name = 'loan_amount'
    prepays = prepays.join(loan_amount)

    #combine E, F & G (there's not many of them)
    prepays['grade'] = prepays['grade'].apply(lambda x: min(x, 'G'))

    prepays = prepays.sort('IssuedDate')
    for d in set(prepays['IssuedDate'].values):
        idx = prepays.IssuedDate == d
        max_mob = prepays.ix[idx, 'MOB'].max()
        prepays.ix[idx, :(max_mob-1)] = prepays.ix[idx, :(max_mob-1)].fillna(0)
        print d, max_mob
        print prepays.ix[idx, :max_mob].sum(0)

    g_prepays = prepays.groupby(['term', 'grade'])

    prepays['low_dti'] = g_prepays['dti'].apply(lambda x: x<x.mean())
    prepays['small_loan'] = g_prepays['loan_amount'].apply(lambda x: x<5000)
g2_prepays = prepays.groupby(['term', 'grade'])
mean_prepays = g2_prepays.mean()

all_grades = list('ABCDEFG')

'''
for t in [36,60]:
    for g in all_grades:
        plt.figure()
        data = (mean_prepays.ix[(t,g,False)] - mean_prepays.ix[(t,g,True)])
        data[:t].plot()
        plt.title('{}, {}'.format(t,g))
        plt.show()

for N in [36, 60]:
    for g in all_grades:
        plt.figure()
        legend = list()
        for i, r in mean_prepays.iterrows():
            if i[0]==N and i[1]==g:
                legend.append(', '.join([str(k) for k in i]))
                plt.plot(r[:N])
        plt.legend(legend, loc=2)
        plt.title(N)
        plt.grid()
        plt.show()

'''

print 'After prepay pivot', (now() - t).total_seconds()
# we are using the empirical first-month prepayment (it's often much higher than 
# nearby months), then we'll smooth the later prepayment rates
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
            prepay_curves['{}{}'.format(i[1], i[0])] = list(smoothed_prepays)
           
            plt.figure()
            plt.plot(empirical_prepays, 'b')
            plt.plot(smoothed_prepays, 'g')
            plt.title(str(i))
            plt.grid()
            plt.show()
            

for term in [36,60]:
    plt.figure()
    for grade in all_grades:
        plt.plot(prepay_curves['{}{}'.format(grade,term)])
    plt.title('Term: {}'.format(term))
    plt.legend(all_grades, loc=2)
    plt.grid()
    plt.show()




#json.dump(prepay_curves, open('/Users/marcshivers/LCModel/prepay_curves.json', 'w'), indent=4)
