from datetime import datetime as dt, timedelta as td
from time import sleep
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import copy
#import production_model as model 
import lclib
import emaillib
import json
import requests
from personalized import p
from session import Session
import utils

log = open(os.path.join(p.parent_dir, 'logfile.txt'), 'a')



class APIHandler(object):

    def __init__(self, account='ira'):
        ''' Valid accounts are 'ira' or 'tax' as defined in personalized.py '''
        self.email = p.get_email(account)
        self.id = p.get_id(account)
        self.key = p.get_key(account)
        self.session = Session(email=self.email) 
        self.session.authenticate()
            
    def get_listed_loans(self, new_only=True):
        loans = []
        try:
            result = requests.get('https://api.lendingclub.com/api/investor/v1/loans/listing', 
                                  headers={'Authorization': self.key},
                                  params={'showAll': not new_only})
            if result.status_code == 200:  #success
                result_js = result.json()
                if 'loans' in result_js.keys():
                    loans = result.json()['loans']
        except:
            pass
        return loans

    def get_cash_balance(self):
        cash = -1
        url = 'https://api.lendingclub.com/api/investor/v1/accounts/{}/availablecash'.format(self.id)
        try:
            result = requests.get(url, headers={'Authorization': self.key})
            if result.status_code == 200:  #success
                result_js = result.json()
                if 'availableCash' in result_js.keys():
                    cash = result.json()['availableCash']
        except:
            pass
        return cash 
    
    def get_notes_owned(self):
        notes = []
        url = 'https://api.lendingclub.com/api/investor/v1/accounts/{}/detailednotes'.format(self.id)
        result = requests.get(url, headers={'Authorization': self.key})
        if result.status_code == 200:  #success
            result_js = result.json()
            if 'myNotes' in result_js.keys():
                notes = result.json()['myNotes']
        return notes


    def stage_order(self, loan_id, amount):
        amount_staged = 0
        payload = {
            'method': 'addToPortfolio',
            'loan_id': loan_id,
            'loan_amount': int(amount),
            'remove': 'false'
        }
        try:
            response = self.session.get('/data/portfolio', query=payload)
            json_response = response.json()
        except:
            log.write('{}: Failed prestage orders\n'.format(dt.now()))
            print '\nFailed to prestage order {}\n'.format(loan_id)
            return 0

        if json_response['result']=='success':
            if 'add_modifications' in json_response.keys():
                mod = json_response['add_modifications']
                if 'loanFractions' in mod.keys():
                    frac = mod['loanFractions']
                    if isinstance(frac, list):
                        frac = frac[0]
                    if isinstance(frac, dict) and 'loanFractionAmountAdded' in frac.keys():
                        amount_staged =  frac['loanFractionAmountAdded'] 
            else:
                amount_staged = amount
        return amount_staged 

    def get_loan_details(self, loan_id):
        '''
        Returns the loan details, including location, current job title, 
        employer, relisted status, and number of inquiries.
        '''
        payload = {
            'loan_id': loan_id
        }
        try:
            response = self.session.post('/browse/loanDetailAj.action', data=payload)
            detail = response.json()
            return detail 
        except:
            return -1

    def get_employer_name(self, loan_id):
        currentCompany = ''
        result = self.get_loan_details(loan_id)
        if isinstance(result, dict):
            currentCompany = utils.only_ascii(result['currentCompany'])
        return currentCompany


class Loan(object):
    def __init__(self, features):
        self.search_time = dt.now()
        self.features = features 
        self.model_outputs = dict()
        self.invested = {
                        'max_stage_amount': 0,
                        'staged_amount': 0
                        }
        self.flags = {
                      'inputs_parsed': False,
                      'api_details_parsed': False,
                      'model_run': False,
                      'details_saved': False,
                      'email_details': False
                      }

 
class Inventory(object):
    def __init__(self):
        self.loans = list()


def get_staged_employer_data(lc, known_loans):
    print '\n'
    for loan in known_loans.values():
        if (loan['staged_amount'] > 0) and (loan['currentCompany'] == ''):
            result = lclib.get_loan_details(lc, loan['id'])
            if isinstance(result, dict):
                loan['currentCompany'] = utils.only_ascii(result['currentCompany'])
                print '{} at {}'.format(loan['currentJobTitle'], loan['currentCompany'])
    print '\n'


def calc_model_sensitivities(loans):
    for loan in loans:
        loancopy = copy.deepcopy(loan)
        loancopy['clean_title_log_odds'] = 0
        loan['dflt_ctlo_zero'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['capitalization_log_odds'] = 0
        loan['dflt_caplo_zero'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['HPA1Yr'] = -5.0 
        loan['dflt_hpa1y_neg5pct'] = model.calc_default_risk(loancopy)        
        loancopy['HPA1Yr'] = 0 
        loan['dflt_hpa1y_zero'] = model.calc_default_risk(loancopy)        
        loancopy['HPA1Yr'] = 5.0
        loan['dflt_hpa1y_5pct'] = model.calc_default_risk(loancopy)        
        loancopy['HPA1Yr'] = 10.0
        loan['dflt_hpa1y_10pct'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['avg_urate'] = 0.03
        loancopy['urate'] = 0.03 
        loan['dflt_urate_3pct'] = model.calc_default_risk(loancopy)        
        loancopy['avg_urate'] = 0.08
        loancopy['urate'] = 0.08 
        loan['dflt_urate_8pct'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['urate_chg'] = 0.02
        loan['dflt_urate_chg_2pct'] = model.calc_default_risk(loancopy)        
        loancopy['urate_chg'] = -0.02
        loan['dflt_urate_chg_neg2pct'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['RENT']
        loan['dflt_rent'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['MORTGAGE']
        loan['dflt_mortgage'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['OWN']
        loan['dflt_own'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['even_loan_amount'] = int(not loan['even_loan_amount']) 
        loan['dflt_not_even_loan_amount'] = model.calc_default_risk(loancopy)        

    return


def add_to_known_loans(known_loans, new_loans):
    new_ids = list()
    for l in new_loans:
        if l['id'] not in known_loans.keys():
            l['max_stage_amount'] = 0
            l['staged_amount'] = 0
            l['default_risk'] = 100
            l['base_irr'] = -100
            l['stress_irr'] = -100
            l['search_time'] = dt.now()
            l['inputs_parsed'] = False 
            l['api_details_parsed'] = False 
            l['model_run'] = False
            l['details_saved'] = False
            l['email_details'] = False
            known_loans[l['id']] = l
            new_ids.append(l['id'])
    return known_loans, new_ids 


def sort_by_model_priority(loans, min_rate):
    return sorted(loans, key=lambda x: (x['int_rate']<=min_rate, np.floor(x['clean_title_log_odds']), x['loanAmount']))


def sort_by_int_rate(loans):
    return sorted(loans, key=lambda x: x['int_rate'])


def get_loans_to_evaluate(known_loans):
    loans_to_evaluate = list()
    for l in known_loans.values():
        if l['api_details_parsed']:
            elapsed = (dt.now()-l['search_time']).total_seconds()
            underfunded = l['max_stage_amount']>l['staged_amount']
            retry_stage = (underfunded==True and elapsed<600)
            retry_model = l['model_run']==False
            if retry_stage or retry_model: 
                loans_to_evaluate.append(l)
    return loans_to_evaluate

def update_recent_loan_info(known_loans, info):
    irr_list = list()
    irr_by_grade = defaultdict(lambda :list())
    irr_data = list()
    for l in known_loans.values():
        if l['base_irr'] > -100:
            elapsed = (dt.now()-l['search_time']).total_seconds()
            if elapsed < 600:
                irr_list.append(l['base_irr']) 
                irr_by_grade[l['grade']].append(l['base_irr'])
                irr_data.append((l['grade'], l['initialListStatus'], l['base_irr']))

    info['average_irr'] = np.mean(irr_list)
    info['irr_by_grade'] = irr_by_grade

    col_names = ['grade', 'initialListStatus', 'base_irr']
    if len(irr_data)==0:
        irr_data = [('A', 'F', -100)]
    info['irr_df'] = pd.DataFrame(data=irr_data, columns=col_names)
    return

def get_loans_to_save(known_loans):
    loans_to_save = list()
    for l in known_loans.values():
        if l['inputs_parsed']==True and l['details_saved']==False and l['currentCompany']!='':
            loans_to_save.append(l)
    return loans_to_save

def get_recently_staged(known_loans):
    recently_staged = list()
    for l in known_loans.values():
        if l['staged_amount'] > 0:
            if 'staged_time' in l.keys():
                if (dt.now()-l['staged_time']).total_seconds() < 3600:
                    recently_staged.append(l)
    return sort_by_int_rate(recently_staged)

def attempt_to_stage(lc, known_loans):
    #Try to stage or restage any loan that didn't get a full allocation
    staged_loans = list()
    for id, l in known_loans.items():
        elapsed = (dt.now() - l['search_time']).total_seconds()
        if (l['staged_amount']<l['max_stage_amount']) and (elapsed < 3600):
            amount_to_invest = l['max_stage_amount'] - l['staged_amount']
            amount_staged = stage_order_fast(lc, l['id'], amount_to_invest)
            if amount_staged > 0:
                l['staged_amount'] += amount_staged
                l['staged_time'] = dt.now()
                cash -= amount_staged 
                staged_loans.append(l)
                print 'Restaged ${} for loan {} for {}'.format(amount_staged, l['id'], l['emp_title']) 
            else:
                print 'Attempted to Restage ${} for {}... FAILED'.format(amount_to_invest, l['emp_title'])

    return staged_loans


def main(min_irr=8, max_invest=500):
    init_dt = dt.now() 
    known_loans = dict()

    login(lc_ira, lc_tax)
    email = 'both of them'
    cash_ira = lc_ira.get_cash_balance()
    cash_tax = lc_tax.get_cash_balance()
    cash_update = dt.now()
    print 'IRA cash balance is ${}'.format(cash_ira)
    print 'Tax cash balance is ${}'.format(cash_tax)
    location_data = lclib.LocationDataManager()

    while True:
        print '\n{}: Checking for new loans'.format(dt.now())
        last_search_loans = get_new_loans_REST()
        known_loans, new_ids = add_to_known_loans(known_loans, last_search_loans) 
        for id in new_ids: 
            model.parse_REST_loan_details(known_loans[id])
        recent_loan_amounts = [l['loanAmount'] for l in known_loans.values() 
                               if l['search_time']>dt.now()-td(minutes=60)]
        num_new_loans = len(recent_loan_amounts)
        value_new_loans = sum(recent_loan_amounts)
        info = {'num_new_loans':num_new_loans, 'value_new_loans':value_new_loans}
        print 'Found {} New Loans.'.format(len(new_ids))
        
        num_newly_staged = 0
        loans_to_evaluate = get_loans_to_evaluate(known_loans)
        loans_to_evaluate = sort_by_model_priority(loans_to_evaluate, min_irr+2)
        for loan in loans_to_evaluate:

            #Parse inputs and add features
            if loan['inputs_parsed']==False:
                location_data.get_zip_features(loan['zip3'])
                loan['inputs_parsed'] = True
            
            if loan['inputs_parsed']==True: 
                #Run risk model
                if loan['model_run']==False:

                    model.calc_default_risk(loan)        
                    model.calc_prepayment_risk(loan)        

                    # calc standard irr
                    irr = model.irr_calculator.calc_irr(loan, loan['default_risk'], loan['prepay_risk'])
                    loan['base_irr'], loan['base_irr_tax'] = irr

                    # calc stressed irr
                    stress_irr = model.irr_calculator.calc_irr(loan, loan['default_max'], loan['prepay_max'])
                    loan['stress_irr'], loan['stress_irr_tax'] = stress_irr

                    utils.invest_amount(loan,  min_irr=min_irr/100., max_invest=max_invest) 

                    # Don't invest in loans that were already passed over as whole loans
                    if loan['initialListStatus']=='W':
                        loan['max_stage_amount'] = 0

                    loan['model_run']=True


                #Stage loan
                amount_to_invest = loan['max_stage_amount'] - loan['staged_amount']
                if amount_to_invest>0:
                    amount_staged = 0
                    if loan['stress_irr_tax'] > 0.04 and loan['grade'] < 'G':
                        amount_staged = stage_order_fast(lc_tax, loan['id'], 500)
                    amount_staged += stage_order_fast(lc_ira, loan['id'], amount_to_invest)

                    loan['staged_amount'] += amount_staged
                    loan['staged_time'] = dt.now()
                    if amount_staged>0: 
                        num_newly_staged += 1
                        loan['email_details'] = True
                    print 'Staged ${} of ${} requested at {}% for {}'.format(loan['staged_amount'], 
                                                                      loan['max_stage_amount'],
                                                                      loan['int_rate'], 
                                                                      loan['emp_title']) 
                                                               
                            
                print '\n{}:\n{}\n'.format(dt.now(), emaillib.detail_str(loan))
        
        if len(new_ids)==0:   # get employer data if no new loans
            update_recent_loan_info(known_loans, info)
            staged_loans = get_recently_staged(known_loans) 
            if np.any([l['email_details'] for l in staged_loans]): 
                print 'Staged Loan Employers:'
                get_staged_employer_data(lc_tax, known_loans)
                calc_model_sensitivities(staged_loans)
                emaillib.email_details(email, staged_loans, info)
            num_recently_staged = len(staged_loans)
            print '{} Loans newly staged; {} recently staged.'.format(num_newly_staged, num_recently_staged)
            print '{} total loans found, valued at ${:1,.0f}.'.format(num_new_loans, value_new_loans)

            print '\n\nRejected Loan Employers:'
            #get_employer_data(lc_tax, known_loans)
            loans_to_save = get_loans_to_save(known_loans) 
            if len(loans_to_save)>0:
                lclib.save_loan_info(loans_to_save)

            # sleep for up to 10 minutes and try again.
            sleep_seconds = utils.sleep_seconds(win_len=1)
            if len(loans_to_evaluate) > 0:
                sleep_seconds = min(sleep_seconds, 30)
            if sleep_seconds > 60:
                try:
                    cash_ira = lc_ira.get_cash_balance()
                    cash_tax = lc_tax.get_cash_balance()
                    cash_update = dt.now()
                    print 'IRA cash balance is ${}'.format(cash_ira)
                    print 'Tax cash balance is ${}'.format(cash_tax)
                except:
                    print '{}: Failed to get cash balance'.format(dt.now()) 
                utils.reset_time()
        else:
            sleep_seconds = 0
            sleep_str = 'No sleeping!'

        '''
        if sleep_seconds > 300:
            sleep_seconds = min(600, sleep_seconds)
            import get_employer_data as ged
            ged.update()
        '''

        sleep_str = '\n{}: Processed all loans... sleeping for {} minutes'.format(dt.now(), sleep_seconds/60.0)
        print sleep_str
        sleep(sleep_seconds)


if __name__=='__main__':
    main()
