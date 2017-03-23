from lendingclub import LendingClub, LendingClubError
from lendingclub.filters import SavedFilter
from datetime import datetime as dt, timedelta as td
from time import sleep
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import copy
import random_forest_model as rfm
import lclib
import json
import requests
import personalized as p

log = open(os.path.join(p.parent_dir, 'logfile.txt'), 'a')


def stage_order_fast(lc, id, amount):
    # Stage each loan
    amount_staged = 0
    payload = {
        'method': 'addToPortfolio',
        'loan_id': id,
        'loan_amount': int(amount),
        'remove': 'false'
    }
    try:
        response = lc.session.get('/data/portfolio', query=payload)
        json_response = response.json()
    except:
        log.write('{}: Failed prestage orders\n'.format(dt.now()))
        print '\nFailed to prestage order {}\n'.format(id)
        return 0

    if json_response['result']=='success':
        if 'add_modifications' in json_response.keys():
            mod = json_response['add_modifications']
            if 'loanFractions' in mod.keys():
                frac = mod['loanFractions']
                if isinstance(frac, list):
                    #print 'Loan Fraction was a list: {}'.format(frac)
                    frac = frac[0]
                if isinstance(frac, dict) and 'loanFractionAmountAdded' in frac.keys():
                    amount_staged =  frac['loanFractionAmountAdded'] 
        else:
            amount_staged = amount
    return amount_staged 


def get_staged_employer_data(lc, known_loans):
    print '\n'
    for loan in known_loans.values():
        if (loan['staged_amount'] > 0) and (loan['currentCompany'] == ''):
            result = lclib.get_loan_details(lc, loan['id'])
            if isinstance(result, dict):
                loan['currentCompany'] = lclib.only_ascii(result['currentCompany'])
                print '{} at {}'.format(loan['currentJobTitle'], loan['currentCompany'])
    print '\n'


def calc_model_sensitivities(loans):
    for loan in loans:
        loancopy = copy.deepcopy(loan)
        loancopy['clean_title_log_odds'] = 0
        loan['dflt_ctlo_zero'] = rfm.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['capitalization_log_odds'] = 0
        loan['dflt_caplo_zero'] = rfm.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['HPA1Yr'] = -5.0 
        loan['dflt_hpa1y_neg5pct'] = rfm.calc_default_risk(loancopy)        
        loancopy['HPA1Yr'] = 0 
        loan['dflt_hpa1y_zero'] = rfm.calc_default_risk(loancopy)        
        loancopy['HPA1Yr'] = 5.0
        loan['dflt_hpa1y_5pct'] = rfm.calc_default_risk(loancopy)        
        loancopy['HPA1Yr'] = 10.0
        loan['dflt_hpa1y_10pct'] = rfm.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['avg_urate'] = 0.03
        loancopy['urate'] = 0.03 
        loan['dflt_urate_3pct'] = rfm.calc_default_risk(loancopy)        
        loancopy['avg_urate'] = 0.08
        loancopy['urate'] = 0.08 
        loan['dflt_urate_8pct'] = rfm.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['urate_chg'] = 0.02
        loan['dflt_urate_chg_2pct'] = rfm.calc_default_risk(loancopy)        
        loancopy['urate_chg'] = -0.02
        loan['dflt_urate_chg_neg2pct'] = rfm.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['RENT']
        loan['dflt_rent'] = rfm.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['MORTGAGE']
        loan['dflt_mortgage'] = rfm.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['OWN']
        loan['dflt_own'] = rfm.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['even_loan_amount'] = int(not loan['even_loan_amount']) 
        loan['dflt_not_even_loan_amount'] = rfm.calc_default_risk(loancopy)        

    return



def get_employer_data(lc, loans):
    print '\n'
    for loan in loans.values():
        if loan['currentCompany'] == '':
            result = lclib.get_loan_details(lc, loan['id'])
            if isinstance(result, dict):
                loan['currentCompany'] = lclib.only_ascii(result['currentCompany'])
                print '{} at {}'.format(loan['currentJobTitle'], loan['currentCompany'])
    print '\n'


def get_new_loans_REST():
    loans = []
    try:
        result = requests.get('https://api.lendingclub.com/api/investor/v1/loans/listing', 
                              headers={'Authorization': p.new_loan_key})
        if result.status_code == 200:  #success
            result_js = result.json()
            if 'loans' in result_js.keys():
                loans = result.json()['loans']
    except:
        pass
    return loans


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


def login(lc_ira, lc_tax):
    import getpass

    pw = getpass.getpass('Password:')
    print '\nLogging in...'
    lc_ira.authenticate(email=p.lc_ira_email, password=pw)
    lc_tax.authenticate(email=p.lc_tax_email, password=pw)


lc_ira = LendingClub()
lc_tax = LendingClub()

def main(min_irr=11, max_invest=500):
    init_dt = dt.now() 
    known_loans = dict()

    login(lc_ira, lc_tax)
    email = 'both of them'
    cash_ira = lc_ira.get_cash_balance()
    cash_tax = lc_tax.get_cash_balance()
    cash_update = dt.now()
    print 'IRA cash balance is ${}'.format(cash_ira)
    print 'Tax cash balance is ${}'.format(cash_tax)
    ExternalData = lclib.ExternalDataManager()

    while True:
        print '\n{}: Checking for new loans'.format(dt.now())
        last_search_loans = get_new_loans_REST()
        known_loans, new_ids = add_to_known_loans(known_loans, last_search_loans) 
        for id in new_ids: 
            rfm.parse_REST_loan_details(known_loans[id])
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
                ExternalData.add_features_to_loan(loan)
                loan['inputs_parsed'] = True
            
            if loan['inputs_parsed']==True: 
                #Run risk model
                if loan['model_run']==False:

                    rfm.calc_default_risk(loan)        
                    rfm.calc_prepayment_risk(loan)        

                    # calc standard irr
                    irr = rfm.IRRCalculator.calc_irr(loan, loan['default_risk'], loan['prepay_risk'])
                    loan['base_irr'], loan['base_irr_tax'] = irr

                    # calc stressed irr
                    stress_irr = rfm.IRRCalculator.calc_irr(loan, loan['default_max'], loan['prepay_max'])
                    loan['stress_irr'], loan['stress_irr_tax'] = stress_irr

                    lclib.invest_amount(loan,  min_irr=min_irr/100., max_invest=max_invest) 

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
                                                               
                            
                print '\n{}:\n{}\n'.format(dt.now(), lclib.detail_str(loan))
        
        if len(new_ids)==0:   # get employer data if no new loans
            update_recent_loan_info(known_loans, info)
            staged_loans = get_recently_staged(known_loans) 
            if np.any([l['email_details'] for l in staged_loans]): 
                print 'Staged Loan Employers:'
                get_staged_employer_data(lc_tax, known_loans)
                calc_model_sensitivities(staged_loans)
                lclib.email_details(email, staged_loans, info)
            num_recently_staged = len(staged_loans)
            print '{} Loans newly staged; {} recently staged.'.format(num_newly_staged, num_recently_staged)
            print '{} total loans found, valued at ${:1,.0f}.'.format(num_new_loans, value_new_loans)

            print '\n\nRejected Loan Employers:'
            get_employer_data(lc_tax, known_loans)
            loans_to_save = get_loans_to_save(known_loans) 
            if len(loans_to_save)>0:
                lclib.save_loan_info(loans_to_save)

            # sleep for up to 10 minutes and try again.
            sleep_seconds = lclib.sleep_seconds(win_len=1)
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
                lclib.reset_time()
            print '{}: Processed all loans... sleeping for {} seconds'.format(dt.now(), sleep_seconds)
        else:
            sleep_seconds = 0

        sleep(sleep_seconds)


if __name__=='__main__':
    main()
