from datetime import datetime as dt, timedelta as td
from time import sleep
import numpy as np
import os
import copy
import lclib
import emaillib
import json
import requests
from personalized import p
from session import Session
import utils
import constants
import invest

log = open(os.path.join(p.parent_dir, 'logfile.txt'), 'a')



model_dir = constants.PathManager.get_dir('training') 
pm = invest.PortfolioManager(model_dir=model_dir, required_return=0.09)
pm.try_for_awhile(1)

  
def main():
    pass

'''
def main(min_irr=8, max_invest=500):
    cash_ira = lc_ira.get_cash_balance()
    cash_tax = lc_tax.get_cash_balance()
    print 'IRA cash balance is ${}'.format(cash_ira)
    print 'Tax cash balance is ${}'.format(cash_tax)

    while True:
        print '\n{}: Checking for new loans'.format(dt.now())
        num_new_loans = len(recent_loan_amounts)
        value_new_loans = sum(recent_loan_amounts)
        info = {'num_new_loans':num_new_loans, 'value_new_loans':value_new_loans}
        print 'Found {} New Loans.'.format(len(new_ids))
        
        num_newly_staged = 0

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

if __name__=='__main__':
    main()

