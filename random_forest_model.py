import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta as td
import json
import os
import sys
from collections import defaultdict
from sklearn.externals import joblib
import lclib

model_hash = 'YMHVDZ'
#model_hash = 'HLKMGF'
model_path = os.path.join(lclib.parent_dir, model_hash)

default_model_file = os.path.join(model_path, 'prod_default_risk_model.pkl')
default_model = joblib.load(default_model_file) 
default_model.verbose=0

prepayment_model_file = os.path.join(model_path, 'prepayment_risk_model.pkl')
prepayment_model = joblib.load(prepayment_model_file) 
prepayment_model.verbose=0


#updated prod model
prod_clean_title_map = json.load(open(os.path.join(model_path, 'prod_clean_title_map.json'),'r'))
prod_clean_title_map = defaultdict(lambda :1e9, prod_clean_title_map )

tok4_cap_title_dict = json.load(open(os.path.join(model_path, 'prod_tok4_capitalization_log_odds.json'),'r'))
tok4_cap_title_dict = defaultdict(lambda :0, tok4_cap_title_dict)
#Note the input x is already in '^{}$' format
cap_title_odds = lambda x: lclib.calc_log_odds(x, tok4_cap_title_dict, 4)
        
tok4_clean_title_dict =json.load(open(os.path.join(model_path, 'prod_tok4_clean_title_log_odds.json'),'r'))
tok4_clean_title_dict = defaultdict(lambda :0, tok4_clean_title_dict)
clean_title_odds = lambda x: lclib.calc_log_odds('^{}$'.format(x), tok4_clean_title_dict, 4)

pctlo_dict = json.load(open(os.path.join(model_path, 'pctlo.json'),'r'))
pctlo_dict = defaultdict(lambda :0, pctlo_dict)
pctlo_odds = lambda x: lclib.calc_log_odds('^{}$'.format(x), pctlo_dict, 4)

def parse_REST_loan_details(dtl):
    try:
        dtl['currentCompany'] = '' #currentCompany data not exposed via API
        if dtl['empTitle']==None: dtl['empTitle'] = 'n/a'
        if dtl['mthsSinceLastDelinq']==None: dtl['mthsSinceLastDelinq'] = lclib.LARGE_INT
        if dtl['mthsSinceLastMajorDerog']==None: dtl['mthsSinceLastMajorDerog'] = lclib.LARGE_INT
        if dtl['mthsSinceLastRecord']==None: dtl['mthsSinceLastRecord'] = lclib.LARGE_INT
        if dtl['empLength']==None: dtl['empLength'] = 0 
        dtl['empTitle'] = lclib.only_ascii(dtl['empTitle']).replace('|', '/')  #for saving the data
        dtl['currentJobTitle'] = dtl['empTitle'].strip()
        
        dtl['emp_title'] = dtl['currentJobTitle'].replace('n/a','Blank')
        dtl['capitalization_title'] = lclib.tokenize_capitalization(dtl['emp_title'])
        dtl['clean_title'] = lclib.clean_title(dtl['emp_title'])
        dtl['clean_short_title'] = dtl['clean_title'][:4]

        dtl['clean_title_rank'] = prod_clean_title_map[dtl['clean_title']]
        dtl['capitalization_log_odds'] = cap_title_odds(dtl['capitalization_title'])
        dtl['clean_title_log_odds'] = clean_title_odds(dtl['clean_title'])
        dtl['pctlo'] = pctlo_odds(dtl['clean_title'])

        dtl['int_rate'] = float(dtl['intRate'])
        dtl['loan_amount'] = float(dtl['loanAmount'])
        dtl['annual_income'] = float(dtl['annualInc'])
        dtl['monthly_payment'] = float(dtl['installment'])
        dtl['revol_bal'] = float(dtl['revolBal'])
        dtl['mod_rate'] = int(10*dtl['int_rate'])-10*int(dtl['int_rate'])
        dtl['loan_term'] = float(dtl['term'])
        dtl['dti'] = float(dtl['dti'])
        dtl['delinq_2yrs'] = float(dtl['delinq2Yrs'])
        dtl['fico_score'] = float(dtl['ficoRangeLow'])
        dtl['zip3'] = float(dtl['addrZip'][:3])
        dtl['state'] = dtl['addrState']
        dtl['inq_last_6mths'] = float(dtl['inqLast6Mths'])
        
        dtl['mths_since_last_major_derog'] = float(dtl['mthsSinceLastMajorDerog'])
        dtl['mths_since_last_delinq'] = float(dtl['mthsSinceLastDelinq'])
        dtl['mths_since_last_record'] = float(dtl['mthsSinceLastRecord'])
        dtl['open_acc'] = float(dtl['openAcc'])
        dtl['pub_rec'] = float(dtl['pubRec'])
        dtl['revol_util'] = float(dtl['revolUtil'])
        dtl['total_acc'] = float(dtl['totalAcc'])
        #dtl['is_inc_verified'] = float(dtl['isIncV']=='VERIFIED')
        dtl['is_inc_verified'] = lclib.api_verification_dict[dtl['isIncV']]

        dtl['monthly_int_payment'] = dtl['loan_amount'] * dtl['int_rate'] / 1200.0 
        earliest_credit = dt.strptime(dtl['earliestCrLine'].split('T')[0],'%Y-%m-%d')
        seconds_per_year = 365*24*60*60.0
        dtl['credit_length'] = (dt.now() - earliest_credit).total_seconds() / seconds_per_year
        dtl['subgrade_number'] = lclib.subgrade_map[dtl['subGrade']]
        dtl['purpose_number'] = lclib.purpose_mapping(dtl['purpose'])
        dtl['emp_length'] = dtl['empLength']/12.0
        dtl['home_status_number'] = lclib.home_map[dtl['homeOwnership'].upper()]
        dtl['even_loan_amount'] = float(dtl['loan_amount'] == np.round(dtl['loan_amount'],-3))
        dtl['revol_bal-loan'] = dtl['revol_bal'] - dtl['loan_amount'] 
        dtl['issue_mth'] = dt.now().month

        assert dtl['annual_income'] > 0, "Income is not positive"
        dtl['loan_pct_income'] = dtl['loan_amount'] / dtl['annual_income']

        dtl['api_details_parsed'] = True

    except:
        print '\n\n\nPARSE FAILED!!!\n\n\n'
    return dtl


def parse_loan_details(dtl):
    try:
        dtl['currentJobTitle'] = lclib.only_ascii(dtl['currentJobTitle'])
        dtl['currentCompany'] = lclib.only_ascii(dtl['currentCompany'])
        dtl['int_rate'] = float(dtl['loanRate'])
        dtl['loan_amount'] = float(dtl['loanAmt'])
        grossIncome = dtl['grossIncome'].replace('n/a','0') 
        dtl['annual_income'] = 12.0*float(grossIncome.split('/')[0].replace('$','').replace(',',''))
        dtl['monthly_payment'] = float(dtl['monthlyPayment'].replace('$','').replace(',',''))
        dtl['revol_bal'] = float(dtl['revolvingCreditBalance'].replace('$','').replace(',',''))
        dtl['emp_title'] = dtl['currentJobTitle'].strip().replace('n/a','Blank')
        dtl['emp_name'] = dtl['currentCompany'].strip().replace('n/a','Blank')
        dtl['capitalization_title'] = lclib.tokenize_capitalization(dtl['emp_title'])
        dtl['clean_title'] = lclib.clean_title(dtl['emp_title'])
        dtl['clean_short_title'] = dtl['clean_title'][:4]
        dtl['mod_rate'] = int(10*dtl['int_rate'])-10*int(dtl['int_rate'])
        dtl['loan_term'] = float(dtl['loanLength'])
        dtl['dti'] = float(dtl['DTI'])
        dtl['delinq_2yrs'] = float(dtl['lateLast2yrs'])
        dtl['fico_score'] = float(dtl['fico'].split('-')[0])
        dtl['zip3'] = float(dtl['location'][:3])
        dtl['state'] = dtl['location'][-2:]
        dtl['inq_last_6mths'] = float(dtl['inquiriesLast6Months'])
        dtl['mths_since_last_delinq'] = float(dtl['monthsSinceLastDelinquency'].replace('n/a','-1'))
        dtl['mths_since_last_record'] = float(dtl['monthsSinceLastRecord'].replace('n/a','-1'))
        dtl['open_acc'] = float(dtl['openCreditLines'])
        dtl['pub_rec'] = float(dtl['publicRecordsOnFile'])
        dtl['revol_util'] = float(dtl['revolvingLineUtilization'].replace('n/a','0').replace('%',''))
        dtl['total_acc'] = float(dtl['totalCreditLines'])
        dtl['is_inc_verified'] = float(dtl['verifiedIncome']=='true')
        dtl['monthly_int_payment'] = dtl['loan_amount'] * dtl['int_rate'] / 1200.0 
        earliest_credit = dt.strptime(dtl['earliestCreditLine'],'%m/%Y')
        seconds_per_year = 365*24*60*60.0
        dtl['credit_length'] = (dt.now() - earliest_credit).total_seconds() / seconds_per_year
        dtl['subgrade_number'] = lclib.subgrade_map[dtl['loanGrade']]
        dtl['purpose_number'] = lclib.purpose_mapping(dtl['purpose'])
        dtl['emp_length'] = lclib.employment_length_map(dtl['completeTenure'])
        dtl['home_status_number'] = lclib.home_map[dtl['homeOwnership'].upper()]
        dtl['even_loan_amount'] = float(dtl['loan_amount'] == np.round(dtl['loan_amount'],-3))
        dtl['revol_bal-loan'] = dtl['revol_bal'] - dtl['loan_amount'] 
        dtl['issue_mth'] = dt.now().month
        dtl['capitalization_log_odds'] = cap_title_odds(dtl['capitalization_title'])
        dtl['clean_title_log_odds'] = clean_title_odds(dtl['clean_title'])
        assert dtl['annual_income']>0

    except:
        return -1
    return dtl


def calc_default_risk(dtl):
    try:
        x = np.zeros(25) * np.nan

        job_title = dtl['emp_title']

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


        x[0] = dtl['loan_amount']
        x[1] = dtl['monthly_payment'] 
        x[2] = dtl['subgrade_number'] 
        x[3] = dtl['purpose_number'] 
        x[4] = dtl['emp_length'] 
        x[5] = dtl['home_status_number']
        x[6] = dtl['annual_income']
        x[7] = dtl['dti']
        x[8] = dtl['revol_util'] 
        x[9] = dtl['total_acc'] 
        x[10] = dtl['credit_length'] 
        x[11] = dtl['even_loan_amount']
        x[12] = dtl['revol_bal-loan'] 
        x[13] = dtl['urate'] 
        x[14] = dtl['annual_income'] / dtl['med_income'] 
        x[15] = dtl['clean_title_rank'] 
        x[16] = dtl['clean_title_log_odds'] 
        x[17] = dtl['capitalization_log_odds']
        x[18] = dtl['monthly_payment'] / dtl['annual_income'] 
        x[19] = dtl['monthly_int_payment'] / dtl['annual_income']
        x[20] = dtl['revol_bal'] / dtl['annual_income'] 
        x[21] = dtl['avg_urate'] 
        x[22] = dtl['urate_chg'] 
        x[23] = dtl['urate_range'] 
        x[24] = dtl['HPA1Yr'] 

        predictions = [tree.predict(x)[0] for tree in default_model.estimators_]
        dtl['default_risk'] = np.mean(predictions) 
        #get bootstrap estimates for default mean estimator 
        #bootstrap_means = [np.mean(np.random.choice(predictions, 100)) for _ in range(1000)]
        #dtl['default_max'] = np.max(bootstrap_means)

        #65th percentile close to the bootstrap max above, but much faster to calculate
        dtl['default_max'] =  np.percentile(predictions, 65)
    
    except Exception as e:
        msg = 'Error in random_forest_model.py::default_risk()'
        msg += '\n{}: Error {} in evaluating random forest\n'.format(dt.now(), str(e))
        msg +='\nInput Loan data:\n'
        for k,v in dtl.iteritems():
            msg += '{}:{}\n'.format(k,v)
        msg += str(sys.exc_info()[1])
        msg += '\n\n'
        print msg
        print zip(range(len(x)), x)
        dtl['default_risk'] = 1.0
        dtl['default_max'] = 1.0
        dtl['default_std'] = 0
   
    return dtl['default_max']



def calc_prepayment_risk(dtl):
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
    try:
        x = np.zeros(23) * np.nan

        job_title = dtl['emp_title']


        x[0] = dtl['loan_amount']
        x[1] = dtl['int_rate'] 
        x[2] = dtl['monthly_payment'] 
        x[3] = dtl['term'] 
        x[4] = dtl['subgrade_number'] 
        x[5] = dtl['purpose_number']
        x[6] = dtl['home_status_number']
        x[7] = dtl['dti']
        x[8] = dtl['inq_last_6mths'] 
        x[9] = dtl['delinq_2yrs'] 
        x[10] = dtl['revol_util'] 
        x[11] = dtl['total_acc']
        x[12] = dtl['credit_length'] 
        x[13] = dtl['even_loan_amount'] 
        x[14] = dtl['revol_bal-loan'] 
        x[15] = dtl['pctlo'] 
        x[16] = dtl['monthly_payment'] / dtl['annual_income'] 
        x[17] = dtl['monthly_int_payment'] / dtl['annual_income'] 
        x[18] = dtl['revol_bal'] / dtl['annual_income'] 
        x[19] = dtl['urate_chg'] 
        x[20] = dtl['HPA1Yr'] 
        x[21] = dtl['fico_score']
        x[22] = dtl['loan_pct_income'] 

        predictions = [tree.predict(x)[0] for tree in prepayment_model.estimators_]
        dtl['prepay_risk'] = np.mean(predictions) 

        #get bootstrap estimates for default mean estimator 
        #bootstrap_means = [np.mean(np.random.choice(predictions, 100)) for _ in range(1000)]
        #dtl['prepay_max'] =  np.max(bootstrap_means)

        #65th percentile close to the bootstrap max above, but much faster to calculate
        dtl['prepay_max'] =  np.percentile(predictions, 65)
    

    except Exception as e:
        msg = 'Error in random_forest_model.py::prepay_risk()'
        msg += '\n{}: Error {} in evaluating random forest\n'.format(dt.now(), str(e))
        msg +='\nInput Loan data:\n'
        for k,v in dtl.iteritems():
            msg += '{}:{}\n'.format(k,v)
        msg += str(sys.exc_info()[1])
        msg += '\n\n'
        print msg
        print zip(range(len(x)), x)
        dtl['prepay_risk'] = 1.0
        dtl['prepay_max'] = 1.0
   
    return dtl['prepay_max']

