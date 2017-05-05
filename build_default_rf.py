import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from collections import defaultdict
import os
import json
from constants import PathManager as paths
from loanstats import load_training_data

# decision variables: 
dv = [
     'accOpenPast24Mths',
     'annualInc',
     'applicationType',
     'avgCurBal',
     'avg_urate',
     'bcOpenToBuy',
     'bcUtil',
     'census_median_income',
     'credit_length',
     'cur_bal-loan_amnt',
     'cur_bal_pct_loan_amnt',
     'dti',
     'empLength',
     'even_loan_amnt',
     'ficoRangeLow',
     'homeOwnership',
     'hpa4',
     'initialListStatus',
     'inqLast6Mths',
     'installment',
     'intRate',
     'int_pct_inc',
     'int_pymt',
     'isIncV',
     'loanAmount',
     'loan_pct_income',
     'moSinOldIlAcct',
     'moSinOldRevTlOp',
     'moSinRcntRevTlOp',
     'moSinRcntTl',
     'mortAcc',
     'mort_bal',
     'mort_pct_credit_limit',
     'mort_pct_cur_bal',
     'mthsSinceLastDelinq',
     'mthsSinceLastMajorDerog',
     'mthsSinceLastRecord',
     'mthsSinceRecentBc',
     'mthsSinceRecentInq',
     'mthsSinceRecentRevolDelinq',
     'numActvBcTl',
     'numActvRevTl',
     'numBcSats',
     'numBcTl',
     'numIlTl',
     'numOpRevTl',
     'numRevAccts',
     'numRevTlBalGt0',
     'numSats',
     'numTlOpPast12m',
     'pctTlNvrDlq',
     'inc_pct_med_inc',
     'percentBcGt75',
     'pubRecBankruptcies',
     'purpose',
     'pymt_pct_inc',
     'revolBal',
     'revolUtil',
     'revol_bal-loan',
     'revol_bal_pct_cur_bal',
     'revol_bal_pct_inc',
     'subGrade',
     'term',
     'totCollAmt',
     'totCurBal',
     'totHiCredLim',
     'totalAcc',
     'totalBalExMort',
     'totalBcLimit',
     'totalIlHighCreditLimit',
     'totalRevHiLim',
     'urate',
     'urate_chg',
     'default_empTitle_shorttoks_odds',
     'prepay_empTitle_shorttoks_odds',
     'empTitle_length',
     'empTitle_frequency',
     ]

def build_default_random_forest():
    print 'Building Default Random Forest'
    df = load_training_data()

    pctl = 65

    iv = '12m_wgt_default'
    extra_cols = [tmp for tmp in [iv, 'issue_d', 'grade', 'term', 'intRate', 'in_sample']
                    if tmp not in dv]
    required_cols = dv + extra_cols

    #Check that all the columns exist:
    for col in required_cols:
        if col not in df.columns:
            print col, 'is not in the cached data'

    fit_data = df.ix[:,required_cols]
    fit_data = fit_data.dropna()

    finite = fit_data.select_dtypes(include=[np.number]).abs().max(1)<np.inf
    fit_data = fit_data.ix[finite]

    fit_data = fit_data.sort('issue_d')
    x_train = fit_data.ix[fit_data.in_sample,:][dv].values
    y_train = fit_data.ix[fit_data.in_sample,:][iv].values
    y_test = fit_data.ix[~fit_data.in_sample,:][iv].values
    x_test = fit_data.ix[~fit_data.in_sample,:][dv].values
    test_int_rate = fit_data.ix[~fit_data.in_sample, 'intRate'].values
    test_term = fit_data.ix[~fit_data.in_sample, 'term'].values


    forest = RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=1000, 
                                   verbose=2, n_jobs=8, oob_score=True, max_features=0.50)
    forest = forest.fit(x_train, y_train) 
    forest.verbose=0

    pf = forest.predict(x_test)
    predictions = [tree.predict(x_test) for tree in forest.estimators_]
    predictions = np.vstack(predictions).T  #loans X trees

    test_data = fit_data.ix[~fit_data.in_sample] 
    test_data['default_prob'] = pf
    test_data['default_prob_65'] = np.percentile(predictions, pctl, axis=1)
      
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
        rate_diff = sample.ix[bottom, 'intRate'].mean() - sample.ix[top, 'intRate'].mean()
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

    data_str += '\n\nOut-of-Bag Score: {}\n'.format(forest.oob_score_)

    print data_str

    time_str = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    fname = os.path.join(paths.get_dir('training'), 'default_forest_{}.txt'.format(time_str))
    with open(fname,'w') as f:
        f.write(data_str)

    pkl_file_name = 'default_randomforest.pkl'
    config = {'inputs': dv, 'pctl': pctl, 'pkl_filename':pkl_file_name, 'feature_name':'default_risk'}
    fname = os.path.join(paths.get_dir('training'), 'default_randomforest.json')
    json.dump(config, open(fname, 'w'), indent=4)

    # pickle the classifier for persistence
    forest_fname = os.path.join(paths.get_dir('training'), pkl_file_name) 
    joblib.dump(forest, forest_fname, compress=3)

