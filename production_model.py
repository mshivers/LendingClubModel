import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta as td
import json
import os
from collections import defaultdict
from sklearn.externals import joblib
import lclib
from personalized import p
import numbers
import utils

model_dir_name = 'Model20170402'
model_path = os.path.join(p.parent_dir, model_dir_name)
model_path = lclib.training_data_dir

class RandomForestModel(object):
    def __init__(self, pkl_file, config_file):
        self.pkl_file = pkl_file
        self.config_file = config_file
        self.random_forest = joblib.load(pkl_file) 
        self.random_forest.verbose=0
        self.config = json.load(open(config_file, 'r'))
        assert 'inputs' in self.config.keys()
        assert 'pctl' in self.config.keys()
 
    def required_inputs(self):
        return self.config['inputs']

    def validate_inputs(self, inputs):
        valid = True
        if not isinstance(inputs, dict):
            print 'RandomForest Input Error: input must be a dict, not a {}'.format(type(inputs))
            valid = False

        for field in self.required_inputs():
            if field in inputs.keys():
                value = inputs[field]
                if not (isinstance(value, numbers.Number) and np.isfinite(value)):
                    print 'RandomForest Invalid Input Error: {}:{} not valid'.format(field, value)
                    valid = False
            else:
                print 'RandomForest Missing Input Error: {} not found'.format(field)
                valid = False

        return valid 

    def run(self, inputs):
        valid = self.validate_inputs(inputs)
        if valid: 
            x = np.zeros(len(self.config['inputs'])) * np.nan
            for i, fld in enumerate(self.config['intputs']):
                x[i] = inputs[fld]

            predictions = [tree.predict(x)[0] for tree in self.random_forest.estimators_]
            prediction =  np.percentile(predictions, self.config['pctl'])
        
        else: 
            prediction = np.nan

        return prediction 




default_model_file = os.path.join(model_path, 'default_risk_model.pkl')
default_model_config_file = os.path.join(model_path, 'default_model_config.json')
default_model = RandomForestModel(default_model_file, default_model_config_file)

prepay_model_file = os.path.join(model_path, 'prepay_risk_model.pkl')
prepay_model_config_file = os.path.join(model_path, 'prepay_model_config.json')
prepay_model = RandomForestModel(prepay_model_file, prepay_model_config_file)

default_curves = json.load(open(os.path.join(model_path, 'default_curves.json'), 'r'))
prepay_curves = json.load(open(os.path.join(model_path, 'prepay_curves.json'), 'r'))
irr_calculator = lclib.ReturnCalculator(default_curves, prepay_curves)

empTitle_freq_config = json.load(open(os.path.join(model_path, 'empTitle_frequency.json'), 'r'))
empTitle_freq_model = lclib.FrequencyModel(**empTitle_freq_config)

'''
prod_clean_title_map = json.load(open(os.path.join(model_path, 'clean_title_rank_map.json'),'r'))
prod_clean_title_map = defaultdict(lambda :1e9, prod_clean_title_map )

caplo_dict = json.load(open(os.path.join(model_path, 'caplo.json'),'r'))
CapitalizationDefault = lclib.LogOddsCalculator(caplo_dict)

ctlo_dict =json.load(open(os.path.join(model_path, 'ctlo.json'),'r'))
TitleDefault = lclib.LogOddsCalculator(ctlo_dict)

pctlo_dict = json.load(open(os.path.join(model_path, 'pctlo.json'),'r'))
TitlePrepay = lclib.LogOddsCalculator(pctlo_dict)
'''
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
        loan['capitalization_log_odds'] = CapitalizationDefault.calc_log_odds(loan['capitalization_title'])

        tokenized_clean_title = '^{}$'.format(loan['clean_title'])
        loan['clean_title_log_odds'] = TitleDefault.calc_log_odds(tokenized_clean_title)
        loan['pctlo'] = TitlePrepay.calc_log_odds(tokenized_clean_title)

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
        'loan_pct_income',
        'delinq_2yrs', 
        'mths_since_last_record',
        'mths_since_last_major_derog',
        'open_acc',
        'pub_rec',
        'revol_bal'
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
        'fico_range_low': 'fico_score',
        'verification_status': 'is_inc_verified',
        'int_pymt': 'monthly_int_payment',
        'med_inc': 'med_income'
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
    elif field=='cur_bal-loan_amnt':
        return loan['totCurBal'] - loan['loan_amount'] 
    elif field=='cur_bal_pct_loan_amnt':
        return loan['totCurBal'] / loan['loan_amount'] 
    else:
        print 'Field {} not found'.format(field)
        return np.nan


