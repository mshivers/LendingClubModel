from lendingclub import LendingClub, LendingClubError
from lendingclub.filters import SavedFilter
from datetime import datetime as dt, timedelta as td
from time import sleep
import numpy as np
import os
import copy
import random_forest_model as rfm
import lclib
import json
import requests

log = open(os.path.join(lclib.parent_dir, 'logfile.txt'), 'a')


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

def stage_orders(lc, loans, amount):
    # Stage each loan
    if isinstance(loans, dict):
        loans = [loans]

    for loan in loans: 
        payload = {
            'method': 'addToPortfolio',
            'loan_id': loan['id'],
            'loan_amount': amount,
            'remove': 'false'
        }
        response = lc.session.get('/data/portfolio', query=payload)
        json_response = response.json()

        # Ensure it was successful before moving on
        if not lc.session.json_success(json_response):
            raise LendingClubError('Could not stage loan {0} on the order: {1}'.format(loan_id, response.text), response)
        else:
            print '{}: Staged ${} for order id = {}'.format(dt.now(), amount, loan['id'])

    #
    # Add all staged loans to the order
    #
    payload = {
        'method': 'addToPortfolioNew'
    }
    response = lc.session.get('/data/portfolio', query=payload)
    json_response = response.json()
    if lc.session.json_success(json_response):
        return True
    else:
        raise LendingClubError('Could not add loans to the order', response.text)


def add_to_order(lc, loan_id, loan_amt):
    '''
    Adds a new investment to the Order.  
    This stages is in Lending Club's Order, hence reserving that amount, which you can cancel later.
    '''
    payload = { 'loan_id': loan_id,
                'loan_amount': loan_amt }
    response = lc.session.post('/browse/addToPortfolio.action', data=payload)
    return response 


def remove_from_portfolio(lc, loan_id):
    # Trying to get this to work; just guessing how to do this...
    payload = { 'method':'removeFromPortfolio',
                'loan_id':loan_id,
                'remove':True}
    response = lc.session.get('/data/portfolio', data=payload)
    return response


def get_newly_listed_loans(new_loans, known_loans):
    return [l['id'] for l in new_loans if l['id'] not in known_loans.keys()]

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
        loan['dflt_ctlo_zero'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['capitalization_log_odds'] = 0
        loan['dflt_caplo_zero'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['HPA1Yr'] = 0
        loan['dflt_hpa1y_zero'] = rfm.default_risk(loancopy)        
        loancopy['HPA1Yr'] = 5.0
        loan['dflt_hpa1y_5pct'] = rfm.default_risk(loancopy)        
        loancopy['HPA1Yr'] = 10.0
        loan['dflt_hpa1y_10pct'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['clean_title_rank'] = 100
        loan['dflt_clean_title_rank_100'] = rfm.default_risk(loancopy)        
        loancopy['clean_title_rank'] = 999999 
        loan['dflt_clean_title_rank_large'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['avg_urate'] = 0.03
        loancopy['urate'] = 0.03 
        loan['dflt_urate_3pct'] = rfm.default_risk(loancopy)        
        loancopy['avg_urate'] = 0.08
        loancopy['urate'] = 0.08 
        loan['dflt_urate_8pct'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['urate_chg'] = 0.02
        loan['dflt_urate_chg_2pct'] = rfm.default_risk(loancopy)        
        loancopy['urate_chg'] = -0.02
        loan['dflt_urate_chg_neg2pct'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['urate_range'] = 0.04
        loan['dflt_urate_range_4pct'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['credit_length'] = 5.0
        loan['dflt_credit_length_5yrs'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['emp_length'] = 0
        loan['dflt_emp_length_zero'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['RENT']
        loan['dflt_rent'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['MORTGAGE']
        loan['dflt_mortgage'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['OWN']
        loan['dflt_own'] = rfm.default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['even_loan_amount'] = int(not loan['even_loan_amount']) 
        loan['dflt_not_even_loan_amount'] = rfm.default_risk(loancopy)        

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
                              headers={'Authorization': 'k1ERom59eg9I39i6ERdotagIlQo='})
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
            l['alpha'] = -100
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
    #emails = {0:('Taxable','marc.shivers@gmail.com'), 1:('IRA','marcshivers@gmail.com')}
    #for k,v in emails.iteritems():
    #    print k, v
    #acct = int(raw_input('Which account do you want to log in to?  '))
    #acct_type, email = emails[acct]
    pw = getpass.getpass('Password:')
    print '\nLogging in...'
    lc_ira.authenticate(email='marcshivers@gmail.com', password=pw)
    lc_tax.authenticate(email='marc.shivers@gmail.com', password=pw)

    #return email, acct_type

lc_ira = LendingClub()
lc_tax = LendingClub()

def main(min_alpha=11, max_invest=500):
    init_dt = dt.now() 
    known_loans = dict()

    login(lc_ira, lc_tax)
    email = 'both of them'
    cash_ira = lc_ira.get_cash_balance()
    cash_tax = lc_tax.get_cash_balance()
    cash_update = dt.now()
    print 'IRA cash balance is ${}'.format(cash_ira)
    print 'Tax cash balance is ${}'.format(cash_tax)
  
    while True:
        print '\n{}: Checking for new loans'.format(dt.now())
        last_search_loans = get_new_loans_REST()
        known_loans, new_ids = add_to_known_loans(known_loans, last_search_loans) 
        for id in new_ids: rfm.parse_REST_loan_details(known_loans[id])
        recent_loan_amounts = [l['loanAmount'] for l in known_loans.values() 
                               if l['search_time']>dt.now()-td(minutes=60)]
        num_new_loans = len(recent_loan_amounts)
        value_new_loans = sum(recent_loan_amounts)
        info = {'num_new_loans':num_new_loans, 'value_new_loans':value_new_loans}
        print 'Found {} New Loans.'.format(len(new_ids))
        
        num_newly_staged = 0
        loans_to_evaluate = get_loans_to_evaluate(known_loans)
        loans_to_evaluate = sort_by_model_priority(loans_to_evaluate, min_alpha+2)
        for loan in loans_to_evaluate:

            #Parse inputs and add features
            if loan['inputs_parsed']==False:
                lclib.add_external_features(loan)
                loan['inputs_parsed'] = True
            
            if loan['inputs_parsed']==True: 
                #Run risk model
                if loan['model_run']==False:
                    loan['default_risk'] = rfm.default_risk(loan)        

                    # the mults below are empirical estimates from 2010 loans 
                    # for total loss percentages as a mult of year 1 defaults.
                    mult = 1.14 if loan['loan_term']==36 else 1.72 
                    loan['alpha'] = loan['int_rate'] - mult * loan['default_risk']
                    loan['max_stage_amount'] = lclib.invest_amount(loan['alpha'], 
                                                                   min_alpha=min_alpha, 
                                                                   max_invest=max_invest) 
                    loan['model_run']=True


                #Stage loan
                amount_to_invest = loan['max_stage_amount'] - loan['staged_amount']
                if amount_to_invest>0:
                    if loan['default_risk'] < 1.00:
                        amount_staged = stage_order_fast(lc_tax, loan['id'], amount_to_invest)
                    else:
                        amount_staged = stage_order_fast(lc_ira, loan['id'], amount_to_invest)

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
