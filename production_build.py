import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta as td
import string
import random
import json
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn import cross_validation
import itertools as it
from collections import Counter, defaultdict
import os
import lclib

file_txt = open(__file__, 'r').readlines()
hash = ''.join(random.sample(string.ascii_uppercase,6))
print 'Hash: {}'.format(hash)
parent_dir = '/home/apprun/LCModel/'
parent_dir = '/Users/marcshivers/LCModel/'
model_dir = '/Users/marcshivers/LCModel/{}'.format(hash)
os.mkdir(model_dir)
with open(os.path.join(model_dir, 'build_file.py'), 'w') as f:
    f.write(''.join(file_txt))


if 'df' not in locals().keys():
    df = lclib.load_training_data()


# decision variables: 
dv = ['loan_amnt', 
      'installment', 
      'sub_grade', 
      'purpose', 
      'emp_length', 
      'home_ownership', 
      'annual_inc', 
      'dti',
      'revol_util', 
      'total_acc', 
      'credit_length',
      'even_loan_amnt', 
      'revol_bal-loan', 
      'urate',
      'pct_med_inc', 
      'clean_title_rank', 
      'ctloC',
      'caploC', 
      'pymt_pct_inc', 
      'int_pct_inc', 
      'revol_bal_pct_inc',
      'avg_urate',
      'urate_chg', 
      'urate_range',
      'hpa4',
    ]


iv = '12m_wgt_default'
extra_cols = [iv, 'issue_d']
if 'int_rate' not in dv:
    extra_cols.append('int_rate')
if 'term' not in dv:
    extra_cols.append('term')
fit_data = df.ix[:,dv+extra_cols]
fit_data = fit_data.dropna()
fit_data = fit_data.sort('issue_d')

oos_cutoff = lclib.oos_cutoff 
isdx = fit_data['issue_d'] < oos_cutoff 
x_train = fit_data.loc[isdx,:][dv].values
y_train = fit_data.loc[isdx,:][iv].values

oos = ~isdx & (fit_data['issue_d']< '2015-06-15')
x_test = fit_data.loc[oos,:][dv].values
y_test = fit_data.loc[oos,:][iv].values

test_int_rate = fit_data.loc[oos]['int_rate'].values
test_term = fit_data.loc[oos]['term'].values

for num_trees in [200]:
    print 'Num Trees = {}'.format(num_trees)
    forest = RandomForestRegressor(n_estimators=num_trees, max_depth=None, 
                                    min_samples_leaf=400, 
                                    verbose=2, n_jobs=8)
    forest = forest.fit(x_train, y_train) 
    forest.verbose=0
    pf = forest.predict(x_test)
    mult = 1.14 * (test_term==36) + 1.72 * (test_term==60)
    exp_loss = pf * mult
    alpha = test_int_rate - 100*exp_loss
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
    data_str += '\n\n{}\t{}\t{}'.format('MinAlpha', 'ROE', 'Count')
    for min_alpha in range(-5, 16):
        portfolio = alpha > min_alpha 
        data_str+= '\n{}\t{:1.2f}\t{}'.format(min_alpha, roe[portfolio].mean(), sum(portfolio))


data_str += '\n\n'
forest_imp = [(dv[i],forest.feature_importances_[i]) for i in forest.feature_importances_.argsort()]
data_str += '\n\nForest Importance\n'
for v in forest_imp:
    data_str += str(v) + '\n'

data_str += '\n\nForest Parameters\n'
for k,v  in forest.get_params().items():
    data_str += '{}: {}\n'.format(k,v)

print data_str
date_str = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
fname = os.path.join(model_dir, 'prod_forest_{}.txt'.format(date_str))
with open(fname,'a') as f:
    f.write(data_str)

# pickle the classifier for persistence
joblib.dump(forest, os.path.join(model_dir, 'prod_default_risk_model.pkl'), compress=9)

#retrieve
#forest = joblib.load('default_risk_forest.pkl')
