import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta as td
import string
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn import cross_validation
import itertools as it
from collections import Counter, defaultdict
import os
import json
from lclib import parent_dir, load_training_data, calc_log_odds, tokenize_capitalization
from lclib import oos_cutoff, construct_loan_dict, calc_npv

if 'df' not in locals().keys():
    df = load_training_data()

pctlo_dict = json.load(open(os.path.join(parent_dir, 'pctlo.json'),'r'))
pctlo = defaultdict(lambda :0, pctlo_dict)
odds_map = lambda x: calc_log_odds('^{}$'.format(x), pctlo, 4)
df['pctlo'] = df['clean_title'].apply(odds_map)

pctloC_dict = json.load(open(os.path.join(parent_dir, 'pctloC.json'),'r'))
pctloC = defaultdict(lambda :0, pctloC_dict)
odds_map = lambda x: calc_log_odds('^{}$'.format(x), pctloC, 4)
df['pctloC'] = df['clean_title'].apply(odds_map)



# decision variables: 
dv = ['loan_amnt', 
      'int_rate', 
      'installment', 
      'term',
      'sub_grade', 
      'purpose', 
      'emp_length', 
      'home_ownership', 
      'annual_inc', 
      'dti',
      'delinq_2yrs', 
      'inq_last_6mths', 
      'mths_since_last_delinq', 
      'mths_since_last_record', 
      'mths_since_last_major_derog',
      'open_acc', 
      'pub_rec', 
      'revol_bal', 
      'revol_util', 
      'total_acc', 
      'verification_status',
      'int_pymt', 
      'credit_length',
      'even_loan_amnt', 
      'revol_bal-loan', 
      'urate',
      'pct_med_inc', 
      'clean_title_rank', 
      'pctlo',
      'pctloC',
      'ctloC',
      'caploC', 
      'pymt_pct_inc', 
      'int_pct_inc', 
      'revol_bal_pct_inc',
      #'zip_code',
      'avg_urate',
      'urate_chg', 
      'urate_range',
      'med_inc', 
      'hpa4',
      'fico_range_low',
      'cur_bal-loan_amnt',
      'cur_bal_pct_loan_amnt'
    ]

iv = '12m_wgt_default'
extra_cols = [tmp for tmp in [iv, 'issue_d', 'grade', 'term', 'int_rate']
                if tmp not in dv]

fit_data = df.ix[:,dv+extra_cols]
fit_data = fit_data.dropna()

oos_cutoff = str(oos_cutoff)
cv_begin = oos_cutoff
cv_end = str(dt(2015,3,1))
print 'OOS Cutoff: {}'.format(oos_cutoff)

fit_data = fit_data.sort('issue_d')
isdx = fit_data['issue_d'] < oos_cutoff 
x_train = fit_data.loc[isdx,:][dv].values
y_train = fit_data.loc[isdx,:][iv].values
oos = (fit_data['issue_d']>= cv_begin) & (fit_data['issue_d']< cv_end) 
y_test = fit_data.loc[oos,:][iv].values
x_test = fit_data.loc[oos,:][dv].values
test_int_rate = fit_data.loc[oos]['int_rate'].values
test_term = fit_data.loc[oos]['term'].values

forest = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=400, verbose=2, n_jobs=8)
forest = forest.fit(x_train, y_train) 
forest.verbose=0
pf = forest.predict(x_test)
mult = 1.14 * (test_term==36) + 1.72 * (test_term==60)
exp_loss = pf * mult
alpha = test_int_rate - 100*exp_loss


test_data = fit_data.loc[oos] 
irr = np.zeros(len(y_test))
for row_num, row_pair in enumerate(test_data.iterrows()):
    row = row_pair[1]
    grade = 'ABCDE'[min(4, row.grade)]
    loan =  construct_loan_dict(grade, row.term, row.int_rate, row.loan_amnt)
    loan['default_risk'] = 100*pf[row_num]
    calc_npv(loan)
    irr[row_num] = loan['irr']

roe = test_int_rate - 100*y_test*mult

titlestr = '{:>8s}'*7 + '\n'
printstr = '{:>8.2f}'*6 + '{:>8.0f}\n'
data_str = 'OOS Cutoff = {}'.format(str(oos_cutoff))
int_ranges = [[0,7],[7,10],[10,12],[12,13.5],[13.5,15],[15,17],[17,20],[20,30]]
for int_range in int_ranges:
    data_str += '\nInt Range: [{},{}]\n'.format(*int_range)
    data_str += titlestr.format('LAlpha','UAlpha','ROE','DExp','DAct','Rate','Num')
    cdx = np.all(zip(test_int_rate>=int_range[0], test_int_rate<=int_range[1]), 1)
    range_alphas = alpha[cdx] 
    pctls = np.arange(0,101,10)
    cutoffs = np.percentile(range_alphas, pctls)
    for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
        idx = np.all(zip(range_alphas>=lower, range_alphas<upper), axis=1)
        empirical_default = 100*(y_test*mult)[cdx][idx].mean()
        model_default =100*(exp_loss[cdx][idx]).mean()
        int_rate = test_int_rate[cdx][idx].mean()
        range_roe = int_rate - empirical_default
        data = (lower,upper, range_roe, model_default,empirical_default, int_rate, sum(idx))
        data_str += printstr.format(*data)

    data_str += titlestr.format('LIRR','UIRR','ROE','DExp','DAct','Rate','Num')
    cdx = np.all(zip(test_int_rate>=int_range[0], test_int_rate<=int_range[1]), 1)
    range_irrs = irr[cdx]
    cutoffs = np.percentile(range_irrs, pctls)
    for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
        idx = np.all(zip(range_irrs>=lower, range_irrs<upper), axis=1)
        empirical_default = 100*(y_test*mult)[cdx][idx].mean()
        model_default =100*(exp_loss[cdx][idx]).mean()
        int_rate = test_int_rate[cdx][idx].mean()
        range_roe = int_rate - empirical_default
        data = (lower,upper, range_roe, model_default,empirical_default, int_rate, sum(idx))
        data_str += printstr.format(*data)

data_str += '\n\n{}\t{}\t{}'.format('MinAlpha', 'ROE', 'Count')
for min_alpha in range(-5, 16):
    portfolio = alpha > min_alpha 
    data_str+= '\n{}\t{:1.2f}\t{}'.format(min_alpha, roe[portfolio].mean(), sum(portfolio))

forest_imp = [(dv[i],forest.feature_importances_[i]) for i in forest.feature_importances_.argsort()]
data_str += '\n\nForest Importance\n'
for v in forest_imp:
    data_str += str(v) + '\n'

data_str += '\n\nForest Parameters\n'
for k,v  in forest.get_params().items():
    data_str += '{}: {}\n'.format(k,v)

print data_str
fname = os.path.join(parent_dir, 'fits/forest_{}.txt'.format(dt.now().strftime('%Y_%m_%d_%H_%M_%S')))
with open(fname,'a') as f:
    f.write(data_str)





# pickle the classifier for persistence
#joblib.dump(forest, 'test_default_risk_model_v2.pkl', compress=3)

#retrieve
#forest = joblib.load('default_risk_forest.pkl')
