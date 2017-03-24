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
from lclib import training_data_dir, load_training_data

if 'df' not in locals().keys():
    df = load_training_data()

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
'''
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
'''
iv = '12m_wgt_default'
extra_cols = [tmp for tmp in [iv, 'issue_d', 'grade', 'term', 'int_rate', 'in_sample']
                if tmp not in dv]

fit_data = df.ix[:,dv+extra_cols]
fit_data = fit_data.dropna()

finite = fit_data.select_dtypes(include=[np.number]).abs().max(1)<np.inf
fit_data = fit_data.ix[finite]

fit_data = fit_data.sort('issue_d')
x_train = fit_data.ix[fit_data.in_sample,:][dv].values
y_train = fit_data.ix[fit_data.in_sample,:][iv].values
y_test = fit_data.ix[~fit_data.in_sample,:][iv].values
x_test = fit_data.ix[~fit_data.in_sample,:][dv].values
test_int_rate = fit_data.ix[~fit_data.in_sample, 'int_rate'].values
test_term = fit_data.ix[~fit_data.in_sample, 'term'].values


forest = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=400, verbose=2, n_jobs=8)
forest = forest.fit(x_train, y_train) 
forest.verbose=0

pf = forest.predict(x_test)
predictions = [tree.predict(x_test) for tree in forest.estimators_]
predictions = np.vstack(predictions).T  #loans X trees

test_data = fit_data.ix[~fit_data.in_sample] 
test_data['default_prob'] = pf
test_data['default_prob_65'] = np.percentile(predictions, 65, axis=1)
  
res_data = list()
grp = test_data.groupby(['grade', 'term'])
default_fld = 'default_prob_65'
for k in sorted(grp.groups.keys(), key=lambda x:(x[1], x[0])):
    sample = grp.get_group(k)
    grp_predict = sample[default_fld]
    pctl10, grp_median, pctl90 = np.percentile(grp_predict.values, [10,50,90])
    bottom = grp_predict<=pctl10
    top = grp_predict>=pctl90
    bottom_default_mean = 100*sample.ix[bottom, iv].mean()
    bottom_predict_mean = 100*grp_predict[bottom].mean()
    top_default_mean = 100*sample.ix[top, iv].mean() 
    top_predict_mean = 100*grp_predict[top].mean()
    rate_diff = sample.ix[bottom, 'int_rate'].mean() - sample.ix[top, 'int_rate'].mean()
    res_data.append([k, len(sample), bottom_default_mean, top_default_mean, 
                     bottom_predict_mean, top_predict_mean, rate_diff])
cols = ['group', 'NObs', 'decile1_actual', 'decile10_actual', 'decile1_predicted', 'decile10_predicted', 
        'rate_diff']
res = pd.DataFrame(res_data, columns=cols)
res['decile1_error'] = res['decile1_predicted'] - res['decile1_actual']
print res

data_str = ''
forest_imp = [(dv[i],forest.feature_importances_[i]) for i in forest.feature_importances_.argsort()]
data_str += '\n\nForest Importance\n'
for v in forest_imp:
    data_str += str(v) + '\n'

data_str += '\n\nForest Parameters\n'
for k,v  in forest.get_params().items():
    data_str += '{}: {}\n'.format(k,v)

data_str += '\n\nDefaults by Grade\n'
data_str += res.to_string()

print data_str
time_str = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
fname = os.path.join(training_data_dir, 'default_forest_{}.txt'.format(time_str))
with open(fname,'w') as f:
    f.write(data_str)

fname = os.path.join(training_data_dir, 'default_variables.txt')
with open(fname,'w') as f:
    f.write('\n'.join(dv))

# pickle the classifier for persistence
forest_fname = os.path.join(training_data_dir, 'default_risk_model_{}.pkl'.format(time_str))
joblib.dump(forest, forest_fname, compress=3)

