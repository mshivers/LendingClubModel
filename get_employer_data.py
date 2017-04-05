from lc import APIHandler
import getpass
import utils
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
from time import sleep
from personalized import p


def random_sleep():
    rand = np.random.randint(0,1000)
    #make it more likely to sleep at night
    hour = dt.now().hour
    if hour <= 5:
        rand += 10  
    else:
        rand -= 1 

    if rand > 999: 
        overnight_time= np.random.choice(np.arange(5))*60*60 + np.random.exponential(1*60*60.)
        print 'Overnight Sleep of {} HOURS'.format(overnight_time/(60*60.))
        sleep(overnight_time)
    else:
        mean = np.random.choice(range(2,6))
        exp_mean = np.random.choice(1.0 / np.linspace(1/600., 2, 1000)) 
        short_time = mean + np.random.exponential(exp_mean)
        if short_time < 60:
            print 'Short sleep of {} seconds'.format(short_time)
        else:
            print 'Short sleep of {} Minutes'.format(short_time/60.)
        tm = 10
        while short_time > tm:
            print '.'
            sleep(tm)
            short_time -= tm
        sleep(short_time)


def update():
    
    account = APIHandler('tax') 

    emp_data_file = os.path.join(p.parent_dir, 'data/loanstats/scraped_data/combined_data.txt')
    remaining_id_file = os.path.join(p.parent_dir, 'data/loanstats/scraped_data/remaining_ids.txt')
    remaining_ids = [int(r) for r in open(remaining_id_file, 'r').read().split('\n')]

    if os.path.exists(emp_data_file):
        existing_ids = set(pd.read_csv(emp_data_file, sep='|', header=None, index_col=0).index)
    else:
        existing_ids = set()

    remaining_ids = [r for r in remaining_ids if r not in existing_ids]
    N = len(remaining_ids)
    with open(emp_data_file, 'a') as f:
        for i, id in enumerate(remaining_ids):
            company = account.get_employer_name(id)
            company = utils.only_ascii(company).replace('|','/')
            if len(company)==0:
                continue
            write_str = '{}|{}'.format(id, company)
            space_str = ' ' * max(0, 50 - len(write_str))
            print N-i, i, write_str, space_str,
            f.write(write_str)
            f.write('\n')
            f.flush()
            random_sleep()

if __name__=='__main__':
    update()
