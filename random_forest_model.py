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
import personalized as p

model_hash = 'YMHVDZ'
#model_hash = 'HLKMGF'
model_path = os.path.join(p.parent_dir, model_hash)

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

def parse_REST_loan_details(loan):
    try:
        loan['currentCompany'] = '' #currentCompany data not exposed via API
        if loan['empTitle']==None: loan['empTitle'] = 'n/a'
        if loan['mthsSinceLastDelinq']==None: loan['mthsSinceLastDelinq'] = lclib.LARGE_INT
        if loan['mthsSinceLastMajorDerog']==None: loan['mthsSinceLastMajorDerog'] = lclib.LARGE_INT
        if loan['mthsSinceLastRecord']==None: loan['mthsSinceLastRecord'] = lclib.LARGE_INT
        if loan['empLength']==None: loan['empLength'] = 0 
        loan['empTitle'] = lclib.only_ascii(loan['empTitle']).replace('|', '/')  #for saving the data
        loan['currentJobTitle'] = loan['empTitle'].strip()
        
        loan['emp_title'] = loan['currentJobTitle'].replace('n/a','Blank')
        loan['capitalization_title'] = lclib.tokenize_capitalization(loan['emp_title'])
        loan['clean_title'] = lclib.clean_title(loan['emp_title'])
        loan['clean_short_title'] = loan['clean_title'][:4]

        loan['clean_title_rank'] = prod_clean_title_map[loan['clean_title']]
        loan['capitalization_log_odds'] = cap_title_odds(loan['capitalization_title'])
        loan['clean_title_log_odds'] = clean_title_odds(loan['clean_title'])
        loan['pctlo'] = pctlo_odds(loan['clean_title'])

        loan['int_rate'] = float(loan['intRate'])
        loan['loan_amount'] = float(loan['loanAmount'])
        loan['annual_income'] = float(loan['annualInc'])
        loan['monthly_payment'] = float(loan['installment'])
        loan['revol_bal'] = float(loan['revolBal'])
        loan['mod_rate'] = int(10*loan['int_rate'])-10*int(loan['int_rate'])
        loan['loan_term'] = float(loan['term'])
        loan['dti'] = float(loan['dti'])
        loan['delinq_2yrs'] = float(loan['delinq2Yrs'])
        loan['fico_score'] = float(loan['ficoRangeLow'])
        loan['zip3'] = float(loan['addrZip'][:3])
        loan['state'] = loan['addrState']
        loan['inq_last_6mths'] = float(loan['inqLast6Mths'])
        
        loan['mths_since_last_major_derog'] = float(loan['mthsSinceLastMajorDerog'])
        loan['mths_since_last_delinq'] = float(loan['mthsSinceLastDelinq'])
        loan['mths_since_last_record'] = float(loan['mthsSinceLastRecord'])
        loan['open_acc'] = float(loan['openAcc'])
        loan['pub_rec'] = float(loan['pubRec'])
        loan['revol_util'] = float(loan['revolUtil'])
        loan['total_acc'] = float(loan['totalAcc'])
        #loan['is_inc_verified'] = float(loan['isIncV']=='VERIFIED')
        loan['is_inc_verified'] = lclib.api_verification_dict[loan['isIncV']]

        loan['monthly_int_payment'] = loan['loan_amount'] * loan['int_rate'] / 1200.0 
        earliest_credit = dt.strptime(loan['earliestCrLine'].split('T')[0],'%Y-%m-%d')
        seconds_per_year = 365*24*60*60.0
        loan['credit_length'] = (dt.now() - earliest_credit).total_seconds() / seconds_per_year
        loan['subgrade_number'] = lclib.subgrade_map[loan['subGrade']]
        loan['purpose_number'] = lclib.purpose_mapping(loan['purpose'])
        loan['emp_length'] = loan['empLength']/12.0
        loan['home_status_number'] = lclib.home_map[loan['homeOwnership'].upper()]
        loan['even_loan_amount'] = float(loan['loan_amount'] == np.round(loan['loan_amount'],-3))
        loan['revol_bal-loan'] = loan['revol_bal'] - loan['loan_amount'] 
        loan['issue_mth'] = dt.now().month

        assert loan['annual_income'] > 0, "Income is not positive"
        loan['loan_pct_income'] = loan['loan_amount'] / loan['annual_income']

        loan['api_details_parsed'] = True

    except:
        print '\n\n\nPARSE FAILED!!!\n\n\n'
    return loan


def parse_loan_details(loan):
    try:
        loan['currentJobTitle'] = lclib.only_ascii(loan['currentJobTitle'])
        loan['currentCompany'] = lclib.only_ascii(loan['currentCompany'])
        loan['int_rate'] = float(loan['loanRate'])
        loan['loan_amount'] = float(loan['loanAmt'])
        grossIncome = loan['grossIncome'].replace('n/a','0') 
        loan['annual_income'] = 12.0*float(grossIncome.split('/')[0].replace('$','').replace(',',''))
        loan['monthly_payment'] = float(loan['monthlyPayment'].replace('$','').replace(',',''))
        loan['revol_bal'] = float(loan['revolvingCreditBalance'].replace('$','').replace(',',''))
        loan['emp_title'] = loan['currentJobTitle'].strip().replace('n/a','Blank')
        loan['emp_name'] = loan['currentCompany'].strip().replace('n/a','Blank')
        loan['capitalization_title'] = lclib.tokenize_capitalization(loan['emp_title'])
        loan['clean_title'] = lclib.clean_title(loan['emp_title'])
        loan['clean_short_title'] = loan['clean_title'][:4]
        loan['mod_rate'] = int(10*loan['int_rate'])-10*int(loan['int_rate'])
        loan['loan_term'] = float(loan['loanLength'])
        loan['dti'] = float(loan['DTI'])
        loan['delinq_2yrs'] = float(loan['lateLast2yrs'])
        loan['fico_score'] = float(loan['fico'].split('-')[0])
        loan['zip3'] = float(loan['location'][:3])
        loan['state'] = loan['location'][-2:]
        loan['inq_last_6mths'] = float(loan['inquiriesLast6Months'])
        loan['mths_since_last_delinq'] = float(loan['monthsSinceLastDelinquency'].replace('n/a','-1'))
        loan['mths_since_last_record'] = float(loan['monthsSinceLastRecord'].replace('n/a','-1'))
        loan['open_acc'] = float(loan['openCreditLines'])
        loan['pub_rec'] = float(loan['publicRecordsOnFile'])
        loan['revol_util'] = float(loan['revolvingLineUtilization'].replace('n/a','0').replace('%',''))
        loan['total_acc'] = float(loan['totalCreditLines'])
        loan['is_inc_verified'] = float(loan['verifiedIncome']=='true')
        loan['monthly_int_payment'] = loan['loan_amount'] * loan['int_rate'] / 1200.0 
        earliest_credit = dt.strptime(loan['earliestCreditLine'],'%m/%Y')
        seconds_per_year = 365*24*60*60.0
        loan['credit_length'] = (dt.now() - earliest_credit).total_seconds() / seconds_per_year
        loan['subgrade_number'] = lclib.subgrade_map[loan['loanGrade']]
        loan['purpose_number'] = lclib.purpose_mapping(loan['purpose'])
        loan['emp_length'] = lclib.employment_length_map(loan['completeTenure'])
        loan['home_status_number'] = lclib.home_map[loan['homeOwnership'].upper()]
        loan['even_loan_amount'] = float(loan['loan_amount'] == np.round(loan['loan_amount'],-3))
        loan['revol_bal-loan'] = loan['revol_bal'] - loan['loan_amount'] 
        loan['issue_mth'] = dt.now().month
        loan['capitalization_log_odds'] = cap_title_odds(loan['capitalization_title'])
        loan['clean_title_log_odds'] = clean_title_odds(loan['clean_title'])
        assert loan['annual_income']>0

    except:
        return -1
    return loan


def calc_model_input_field(loan, field):
    ''' The historical data the models were trained on contain different fields than the data
    the real-time API returns, so they need to be translated.  This function does that'''

    # first return any field that doesn't require translation
    unchanged_fields = [
        'emp_length', 
        'avg_urate', 
        'urate_chg', 
        'urate_range', 
        'dti', 
        'revol_util', 
        'total_acc', 
        'credit_length', 
        'revol_bal-loan', 
        'urate',
        'clean_title_rank',
        'int_rate',
        'term',
        'inq_last_6mths', 
        'total_acc', 
        'credit_length',
        'pctlo',
        'urate_chg', 
        'loan_pct_income'
        ]
    if field in unchanged_fields:
        return loan[field]

    # next process fields that are just renamed, but don't require any calculation
    renamed_fields = {
        'loan_amnt':'loan_amount',
        'installment': 'monthly_payment',
        'sub_grade': 'subgrade_number',
        'purpose': 'purpose_number',
        'home_ownership': 'home_status_number',
        'annual_inc': 'annual_income',
        'even_loan_amnt': 'even_loan_amount', 
        'ctloC': 'clean_title_log_odds',
        'caploC': 'capitalization_log_odds', 
        'hpa4': 'HPA1Yr',
        'mths_since_last_delinq': 'delinq_2yrs',
        'fico_range_low': 'fico_score'
        }
    if field in renamed_fields.keys():
        return loan[renamed_fields[field]]

    # finally, process those fields that require a calculation
    if field=='pymt_pct_inc':
        return loan['monthly_payment'] / loan['annual_income'] 
    elif field=='int_pct_inc':
        return loan['monthly_int_payment'] / loan['annual_income'] 
    elif field=='revol_bal_pct_inc':
        return loan['revol_bal'] / loan['annual_income'] 
    elif field=='pct_med_inc':
        return loan['annual_income'] / loan['med_income']
    else:
        return np.nan


def calc_default_risk(loan):
    try:
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

        x = np.zeros(len(dv)) * np.nan
        for i, fld in enumerate(dv):
            x[i] = calc_model_input_field(loan, fld)

        predictions = [tree.predict(x)[0] for tree in default_model.estimators_]
        loan['default_risk'] = np.mean(predictions) 

        # 65th percentile more closely matches the OOS realization
        loan['default_max'] =  np.percentile(predictions, 65)
    
    except Exception as e:
        msg = 'Error in random_forest_model.py::default_risk()'
        msg += '\n{}: Error {} in evaluating random forest\n'.format(dt.now(), str(e))
        msg +='\nInput Loan data:\n'
        for k,v in loan.iteritems():
            msg += '{}:{}\n'.format(k,v)
        msg += str(sys.exc_info()[1])
        msg += '\n\n'
        print msg
        print zip(range(len(x)), x)
        loan['default_risk'] = 1.0
        loan['default_max'] = 1.0
     
    return loan['default_max']



def calc_prepayment_risk(loan):
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
        x = np.zeros(len(dv)) * np.nan
        for i, fld in enumerate(dv):
            x[i] = calc_model_input_field(loan, fld)
    
        predictions = [tree.predict(x)[0] for tree in prepayment_model.estimators_]
        loan['prepay_risk'] = np.mean(predictions) 
        loan['prepay_max'] =  np.percentile(predictions, 65)
    

    except Exception as e:
        msg = 'Error in random_forest_model.py::prepay_risk()'
        msg += '\n{}: Error {} in evaluating random forest\n'.format(dt.now(), str(e))
        msg +='\nInput Loan data:\n'
        for k,v in loan.iteritems():
            msg += '{}:{}\n'.format(k,v)
        msg += str(sys.exc_info()[1])
        msg += '\n\n'
        print msg
        print zip(range(len(x)), x)
        loan['prepay_risk'] = 1.0
        loan['prepay_max'] = 1.0
   
    return loan['prepay_max']

