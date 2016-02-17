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
from lclib import oos_cutoff

if 'df' not in locals().keys():
    df = load_training_data()

if 'urate' not in df.columns: 
    urate = pd.read_csv(os.path.join(parent_dir, 'urate_by_3zip.csv'), index_col=0)
    ur = pd.DataFrame(np.zeros((len(urate),999))*np.nan,index=urate.index, columns=[str(i) for i in range(1,1000)])
    ur.ix[:,:] = urate.median(1).values[:,None]
    ur.ix[:,urate.columns] = urate
    avg_ur = pd.rolling_mean(ur, 12)
    ur_chg = ur - ur.shift(12)
    
    hpa4 = pd.read_csv(os.path.join(parent_dir, 'hpa4.csv'), index_col = 0)
    mean_hpa4 = hpa4.mean(1)
    missing_cols = [str(col) for col in range(0,1000) if str(col) not in hpa4.columns]
    for c in missing_cols:
        hpa4[c] = mean_hpa4

    z2mi = json.load(open('zip2median_inc.json','r'))
    z2mi = dict([(int(z), float(v)) for z,v in zip(z2mi.keys(), z2mi.values())])
    z2mi = defaultdict(lambda :np.mean(z2mi.values()), z2mi)

    # separate out employer name (before 9/23/2013) from employment title (after that)
    def return_word(x, position=0):
        if len(x)==0:
            return ''
        else:
            return x.split()[position]
    df['title_capitalization'] = df['emp_title'].apply(tokenize_capitalization)

    clean_title_count = Counter(df['clean_title'].values)
    clean_titles_sorted = [ttl[0] for ttl in sorted(clean_title_count.items(), key=lambda x:-x[1])]
    clean_title_map = dict(zip(clean_titles_sorted, range(len(clean_titles_sorted))))

    # process job title features
    df['clean_title_rank'] = df['clean_title'].apply(lambda x:clean_title_map[x])
  
    one_year = 365*24*60*60*1e9
    df['credit_length'] = ((df['issue_d'] - df['earliest_cr_line']).astype(int)/one_year)
    df['credit_length'] = df['credit_length'].apply(lambda x: max(-1,round(x,0)))
    df['even_loan_amnt'] = df['loan_amnt'].apply(lambda x: float(x==round(x,-3)))
    df['int_pymt'] = df['loan_amnt'] * df['int_rate'] / 1200.0
    df['loan_amnt'] = df['loan_amnt'].apply(lambda x: round(x,-3))
    df['revol_bal-loan'] = df['revol_bal'] - df['loan_amnt']

    df['urate_d'] = df['issue_d'].apply(lambda x: int(str((x-td(days=60)))[:7].replace('-','')))
    df['urate'] = [ur[a][b] for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['avg_urate'] = [avg_ur[a][b] for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['urate_chg'] = [ur_chg[a][b] for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['max_urate'] = [ur[a][:b].max() for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['min_urate'] = [ur[a][:b].min() for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['urate_range'] = df['max_urate'] - df['min_urate'] 

    df['issue_mth'] = df['issue_d'].apply(lambda x:int(str(x)[5:7]))
    df['med_inc'] = df['zip_code'].apply(lambda x:z2mi[x])
    df['pct_med_inc'] = df['annual_inc'] / df['med_inc']


    df['hpa_date'] = df['issue_d'].apply(lambda x:x-td(days=120))
    df['hpa_qtr'] = df['hpa_date'].apply(lambda x: 100*x.year + x.month/4 + 1)
    hpa4 = pd.read_csv(os.path.join(parent_dir, 'hpa4.csv'), index_col = 0)
    missing_cols = [str(col) for col in range(0,1000) if str(col) not in hpa4.columns]
    mean_hpa4 = hpa4.mean(1)
    for c in missing_cols:
        hpa4[c] = mean_hpa4
    df['hpa4'] = [hpa4.ix[a,b] for a,b in zip(df['hpa_qtr'], df['zip_code'].apply(lambda x: str(int(x))))]

    df['pymt_pct_inc'] = df['installment'] / df['annual_inc'] 
    df['revol_bal_pct_inc'] = df['revol_bal'] / df['annual_inc']
    df['int_pct_inc'] = df['int_pymt'] / df['annual_inc'] 

    # This is estimated using only loans of grede C or lower
    df['title_capitalization'] = df['emp_title'].apply(tokenize_capitalization)
    ctloC_dict = json.load(open('ctloC.json','r'))
    ctloC = defaultdict(lambda :0, ctloC_dict)
    odds_map = lambda x: calc_log_odds('^{}$'.format(x), ctloC, 4)
    df['ctloC'] = df['clean_title'].apply(odds_map)

    caploC = json.load(open('caploC.json','r'))
    caploC = defaultdict(lambda :0, caploC)
    odds_map = lambda x: calc_log_odds(x, caploC, 4)
    df['caploC'] = df['title_capitalization'].apply(odds_map)
    
df['cur_bal-loan_amnt'] = df['tot_cur_bal'] - df['loan_amnt'] 
df['cur_bal_pct_loan_amnt'] = df['tot_cur_bal'] / df['loan_amnt'] 


# decision variables: 
dv = ['loan_amnt', 
      #'int_rate', 
      'installment', 
      #'term',
      'sub_grade', 
      'purpose', 
      'emp_length', 
      'home_ownership', 
      'annual_inc', 
      'dti',
      #'delinq_2yrs', 
      #'inq_last_6mths', 
      #'mths_since_last_delinq', 
      #'mths_since_last_record', 
      #'mths_since_last_major_derog',
      #'open_acc', 
      #'pub_rec', 
      #'revol_bal', 
      'revol_util', 
      'total_acc', 
      #'verification_status',
      #'int_pymt', 
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
      #'avg_urate',
      'urate_chg', 
      #'urate_range',
      #'med_inc', 
      'hpa4',
      #'fico_range_low',
      #'cur_bal-loan_amnt',
      #'cur_bal_pct_loan_amnt'
    ]

iv = '12m_wgt_default'
extra_cols = [iv, 'issue_d']
if 'int_rate' not in dv:
    extra_cols.append('int_rate')
if 'term' not in dv:
    extra_cols.append('term')
fit_data = df.ix[:,dv+extra_cols]
fit_data = fit_data.dropna()


cv_begin = oos_cutoff
cv_end = dt(2015,3,1)
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
