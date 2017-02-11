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

# decision variables: 
dv = [
      'loan_amnt', 
      'int_rate', 
      'installment', 
      'term',
      'sub_grade', 
      'purpose', 
      'home_ownership', 
      'dti',
      'inq_last_6mths', 
      'mths_since_last_delinq', 
      'revol_util', 
      'total_acc', 
      'credit_length',
      'even_loan_amnt', 
      'revol_bal-loan', 
      'pctlo',
      'pymt_pct_inc', 
      'int_pct_inc', 
      'revol_bal_pct_inc',
      'urate_chg', 
      'hpa4',
      'fico_range_low',
      'loan_pct_income',
    ]

iv = '12m_prepay'
extra_cols = [tmp for tmp in [iv, 'loan_status', 'mob', 'issue_d', 'grade', 'term', 'int_rate']
                if tmp not in dv]

fit_data = df.ix[:,dv+extra_cols]
fit_data = fit_data.dropna()

oos_cutoff = str(oos_cutoff)
cv_begin = oos_cutoff
cv_end = str(dt(2016,3,1))
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
predicted_prepayment = forest.predict(x_test)


titlestr = '{:>8s}'*6 + '\n'
printstr = '{:>8.2f}'*5 + '{:>8.0f}\n'
data_str = 'OOS Cutoff = {}\n\n'.format(str(oos_cutoff))
pctls = np.arange(0,101,10)
cutoffs = np.percentile(predicted_prepayment, pctls)
paid = fit_data.ix[oos, 'loan_status']=='Fully Paid'
sse=0.0
data_str += titlestr.format('LPrepay','UPrepay','PExp','PAct','Rate','Num')
for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
    cdx = np.all(zip(predicted_prepayment>=lower, predicted_prepayment<=upper), 1)
    range_prepayments = predicted_prepayment[cdx] 
    empirical_prepay = 100*y_test[cdx].mean()
    model_prepay =100*range_prepayments.mean()
    int_rate = test_int_rate[cdx].mean()
    data = (100*lower,100*upper, model_prepay,empirical_prepay, int_rate, sum(cdx))
    sse += (empirical_prepay - model_prepay)**2
    data_str += printstr.format(*data)

rmse = np.sqrt(sse/(len(pctls)-1))
print 'RMSE = {}'.format(rmse)

data_str += '\n\n{}\t{}\t{}'.format('MinPrepay', 'AvgLowPrepay',  'AvgHighPrepay')
for min_prepay in range(5, 31):
    portfolio = predicted_prepayment > min_prepay/100.
    data_str+= '\n{}\t{:1.2f}\t{:1.2f}'.format(min_prepay, 100*y_test[portfolio].mean(),100*y_test[~portfolio].mean())

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
joblib.dump(forest, 'prepayment_risk_model.pkl', compress=3)

