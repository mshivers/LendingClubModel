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

parent_dir = '/home/apprun/LCModel/'
parent_dir = '/Users/marcshivers/LCModel/'
model_dir = '/Users/marcshivers/LCModel/{}'.format(hash)
os.mkdir(model_dir)
with open(os.path.join(model_dir, 'build_file.py'), 'w') as f:
    f.write(''.join(file_txt))


if 'df' not in locals().keys():
    df = lclib.load_training_data()

    urate = pd.read_csv(os.path.join(parent_dir, 'urate_by_3zip.csv'), index_col=0)
    ur = pd.DataFrame(np.zeros((len(urate),999))*np.nan,index=urate.index, columns=[str(i) for i in range(1,1000)])
    ur.ix[:,:] = urate.median(1).values[:,None]
    ur.ix[:,urate.columns] = urate
    avg_ur = pd.rolling_mean(ur, 12)
    ur_chg = ur - ur.shift(12)

    hpa4 = pd.read_csv(os.path.join(parent_dir, 'hpa4.csv'), index_col = 0)
    missing_cols = [str(col) for col in range(0,1000) if str(col) not in hpa4.columns]
    mean_hpa4 = hpa4.mean(1)
    for c in missing_cols:
        hpa4[c] = mean_hpa4

    z2mi = json.load(open(os.path.join(parent_dir, 'zip2median_inc.json'),'r'))
    z2mi = dict([(int(z), float(v)) for z,v in zip(z2mi.keys(), z2mi.values())])
    z2mi = defaultdict(lambda :np.mean(z2mi.values()), z2mi)

    df['title_first_word'] = df['clean_title'].apply(lambda x:x.split()[0])
    df['title_last_word'] = df['clean_title'].apply(lambda x:x.split()[-1])
    df['short_title'] = df['clean_title'].apply(lambda x:x[:4])

    ttl_words = Counter([word for ttl in df['clean_title'].values 
                                for word in str(ttl).split()])
    ttl_words_sorted = sorted(ttl_words.items(), key=lambda x:-x[1])
    ttl_word_map = dict(zip([w[0] for w in ttl_words_sorted], range(len(ttl_words_sorted))))
    json.dump(ttl_word_map, open(os.path.join(model_dir, 'prod_clean_title_words_map.json'),'w'))

    short_title_count = Counter(df['short_title'].values)
    short_titles_sorted = [ttl[0] for ttl in sorted(short_title_count.items(), key=lambda x:-x[1])]
    short_title_map = dict(zip(short_titles_sorted, range(len(short_titles_sorted))))
    json.dump(short_title_map, open(os.path.join(model_dir, 'prod_clean_short_title_map.json'),'w'))

    clean_title_count = Counter(df['clean_title'].values)
    clean_titles_sorted = [ttl[0] for ttl in sorted(clean_title_count.items(), key=lambda x:-x[1])]
    clean_title_map = dict(zip(clean_titles_sorted, range(len(clean_titles_sorted))))
    json.dump(clean_title_map, open(os.path.join(model_dir, 'prod_clean_title_map.json'),'w'))

    emp_title_count = Counter(df['emp_title'].values)
    emp_titles_sorted = [ttl[0] for ttl in sorted(emp_title_count.items(), key=lambda x:-x[1])]
    emp_title_map = dict(zip(emp_titles_sorted, range(len(emp_titles_sorted))))
    json.dump(emp_title_map, open(os.path.join(model_dir, 'prod_emp_title_map.json'),'w'))

    # process job title features
    df['emp_title_rank'] = df['emp_title'].apply(lambda x:emp_title_map[x])
    df['title_first_word_rank'] = df['title_first_word'].apply(lambda x:ttl_word_map[x])
    df['title_last_word_rank'] = df['title_first_word'].apply(lambda x:ttl_word_map[x])
    df['clean_title_rank'] = df['clean_title'].apply(lambda x:clean_title_map[x])
    df['short_title_rank'] = df['short_title'].apply(lambda x:short_title_map[x])

    df['title_len'] = df['emp_title'].apply(lambda x:len(str(x)))
    df['title_num_words'] = df['emp_title'].apply(lambda x:str(x).count(' '))
    df['title_upper_pct'] = df['emp_title'].apply(lambda x:np.mean([l.isupper() for l in str(x)]))
    df['title_upper_count'] = df['emp_title'].apply(lambda x:np.sum([l.isupper() for l in str(x)]))

    one_year = 365*24*60*60*1e9
    df['credit_length'] = ((df['issue_d'] - df['earliest_cr_line']).astype(int)/one_year).apply(lambda x: max(-1,x))
    df['even_loan_amnt'] = df['loan_amnt'].apply(lambda x: float(x==round(x,-3)))
    df['revol_bal-loan'] = df['revol_bal'] - df['loan_amnt']
    df['int_pymt'] = df['loan_amnt'] * df['int_rate'] / 1200.0
    df['revol_bal_pct_inc'] = df['revol_bal'] / df['annual_inc']
    
    df['urate_d'] = df['issue_d'].apply(lambda x: int(str((x-td(days=60)))[:7].replace('-','')))
    df['urate'] = [ur[a][b] for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['avg_urate'] = [avg_ur[a][b] for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['urate_chg'] = [ur_chg[a][b] for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['max_urate'] = [ur[a][:b].max() for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['min_urate'] = [ur[a][:b].min() for a,b in zip(df['zip_code'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['urate_range'] = df['max_urate'] - df['min_urate'] 

    df['issue_mth'] = df['issue_d'].apply(lambda x:int(str(x)[5:7]))

    df['mod_rate'] = df['int_rate'].apply(lambda x:(int(10*x)-10*int(x)))
    df['med_inc'] = df['zip_code'].apply(lambda x:z2mi[x])
    df['pct_med_inc'] = df['annual_inc'] / df['med_inc']
    df['pymt_pct_inc'] = df['installment'] / df['annual_inc'] 

    df['hpa_date'] = df['issue_d'].apply(lambda x:x-td(days=120))
    df['hpa_qtr'] = df['hpa_date'].apply(lambda x: 100*x.year + x.month/4 + 1)
    df['hpa4'] = [hpa4.ix[a,b] for a,b in zip(df['hpa_qtr'], df['zip_code'].apply(lambda x: str(int(x))))]

    tok4_clean_title_log_odds_dict = json.load(open('ctlo.json','r'))
    json.dump(tok4_clean_title_log_odds_dict, open(os.path.join(model_dir, 
        'prod_tok4_clean_title_log_odds_dict.json'),'w'))
    tok4_clean_title_log_odds_dict = defaultdict(lambda :0, tok4_clean_title_log_odds_dict)
    odds_map = lambda x: lclib.calc_log_odds('^{}$'.format(x), tok4_clean_title_log_odds_dict, 4)
    df['tok4_clean_title_log_odds'] = df['clean_title'].apply(odds_map)

    df['title_capitalization'] = df['emp_title'].apply(lclib.tokenize_capitalization)
    tok4_capitalization_log_odds_dict = json.load(open('caplo.json','r'))
    json.dump(tok4_capitalization_log_odds_dict, open(os.path.join(model_dir, 
        'prod_tok4_capitalization_log_odds_dict.json'),'w'))
    tok4_capitalization_log_odds_dict = defaultdict(lambda :0, tok4_capitalization_log_odds_dict)
    odds_map = lambda x: lclib.calc_log_odds(x, tok4_capitalization_log_odds_dict, 4)
    df['tok4_capitalization_log_odds'] = df['title_capitalization'].apply(odds_map)

    fit_data = df.fillna(-1)

# decision variables: 
dv = ['loan_amnt', 'int_rate', 'installment', 'term', 'sub_grade', 'purpose', 
      'emp_length', 'home_ownership', 'annual_inc', 'dti',
      'delinq_2yrs', 
      'fico_range_low',
      'inq_last_6mths', 
      'mths_since_last_delinq', 'mths_since_last_record',
      'open_acc', 
      'pub_rec', 
      'zip_code',
      'revol_bal', 'revol_util', 'total_acc', 'is_inc_v',
      'int_pymt', 
      'credit_length',
      'even_loan_amnt', 'revol_bal-loan', 
      'urate','avg_urate','urate_chg',
      'mod_rate',
      'med_inc', 'pct_med_inc', 
      'emp_title_rank',
      'clean_title_rank', 
      'short_title_rank',
      'tok4_clean_title_log_odds', 
      'tok4_capitalization_log_odds', 
      'pymt_pct_inc', 
      'hpa4',
      'revol_bal_pct_inc'
        ]

iv = '12m_wgt_default'

fit_data = fit_data.sort('issue_d')

oos_cutoff = dt(2014,3,1) 
isdx = fit_data['issue_d'] < oos_cutoff 
x_train = fit_data.loc[isdx,:][dv].values
y_train = fit_data.loc[isdx,:][iv].values

oos = ~isdx & (fit_data['issue_d']<dt(2014,5,1))
x_test = fit_data.loc[oos,:][dv].values
y_test = fit_data.loc[oos,:][iv].values



for num_trees in [400]:
    print 'Num Trees = {}'.format(num_trees)
    forest = RandomForestRegressor(n_estimators=num_trees, max_depth=None, 
                                    compute_importances=True, 
                                    min_samples_leaf=100, verbose=2, n_jobs=-1)
    forest = forest.fit(x_train, y_train) 
    forest.verbose=0
    pf = forest.predict(x_test)
    mult = 1.14 * (x_test[:,3]==36) + 1.72 * (x_test[:,3]==60)
    exp_loss = pf * mult
    alpha = x_test[:,1] - 100*exp_loss
    roe = x_test[:,1] - 100*y_test*mult

    titlestr = '{:>8s}'*7 + '\n'
    printstr = '{:>8.2f}'*6 + '{:>8.0f}\n'
    data_str = 'OOS Cutoff = {}'.format(str(oos_cutoff))
    int_ranges = [[0,7],[7,10],[10,12],[12,13.5],[13.5,15],[15,17],[17,20],[20,30]]
    for int_range in int_ranges:
        data_str += '\nInt Range: [{},{}]\n'.format(*int_range)
        data_str += titlestr.format('LAlpha','UAlpha','ROE','DExp','DAct','Rate','Num')
        cdx = np.all(zip(x_test[:,1]>=int_range[0], x_test[:,1]<=int_range[1]), 1)
        range_alphas = alpha[cdx] 
        pctls = np.arange(0,101,10)
        cutoffs = np.percentile(range_alphas, pctls)
        for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
            idx = np.all(zip(range_alphas>=lower, range_alphas<upper), axis=1)
            empirical_default = 100*(y_test*mult)[cdx][idx].mean()
            model_default =100*(exp_loss[cdx][idx]).mean()
            int_rate = x_test[cdx,1][idx].mean()
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
