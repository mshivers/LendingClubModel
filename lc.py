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
                utils.save_loan_info(loans_to_save)

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
