from lclib import LendingClub 
import getpass
import utils
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from time import sleep
from personalized import p
from constants import color

class PausedAccount(LendingClub):
    url = 'https://www.lendingclub.com/browse/browse.action'
    url2 = 'https://www.lendingclub.com/foliofn/tradingInventory.action'
    def __init__(self, account):
        LendingClub.__init__(self, account)
        sleep(2)
        self.session.get(self.url)
        self.set_next_call()
        self.auth_count = 1
        self.name_count = 0

    def set_next_call(self):
        sleep_seconds = self.sleep_time()
        self.next_call = dt.now() + td(seconds=sleep_seconds)

    def sleep_time(self):
        mean = 6.0 
        exp_mean = np.random.choice(1.0 / np.linspace(1/5., 1, 10)) 
        sleep_time = mean + np.random.exponential(exp_mean)
        return sleep_time 
 
    def seconds_to_wake(self):
        return max(0, (self.next_call - dt.now()).total_seconds())

    def get_name(self, id):
        self.name_count += 1
        if dt.now() < self.next_call:
            wait = self.seconds_to_wake()
            sleep(wait)
        name = self.get_employer_name(id)
        self.set_next_call()
        if name == None:
            sleep(5)
            self.session.get(self.url)
            self.auth_count += 1
            wait = self.seconds_to_wake()
            sleep(wait)
            name = self.get_employer_name(id)
            self.set_next_call()
        return name

def sleep_func(tm):
    incr = 30 
    while tm > incr:
        sleep(incr)
        minutes_remaining = tm/60
        if minutes_remaining>1:
            print '{:1.0f} Minutes to go'.format(minutes_remaining)
        else:
            print '.'
        tm -= incr 
    sleep(tm)

def acc_sleep_string(accounts, last_acc):
    def to_str(x):
        if x<60:
            tm_str = '{:1.1f} sec'.format(x)
        else:
            tm_str = '{:1.0f} Min'.format(x/60)
        return tm_str

    out = ''
    wake = 999999
    tms = [(acc.account_type, acc.seconds_to_wake(), acc) for acc in accounts]
    for acc_type, slp, acc in tms:
        incr_str = '{}: {}'.format(acc_type, to_str(slp))
        trailing_spaces = ' ' * max(0, 16 - len(incr_str))
        incr_str += trailing_spaces
        #if acc==last_acc:
        #    incr_str = color.RED + color.BOLD + incr_str + color.END
        out += incr_str 
    
    wake = min(tms, key=lambda x: x[1])
    out += 'Waking in {} for {}'.format(to_str(wake[1]), wake[0])
    return out


def update(max_num=1000, max_fails=3):
    if max_num is None:
        max_num=99999
    acct_names = np.random.choice(['tax', 'ira', 'hjg'], 1, replace=True) 

    emp_data_file = os.path.join(p.parent_dir, 'data/loanstats/scraped_data/SCRAPE_FILE.txt')
    emp_data = pd.read_csv(emp_data_file, sep='|', header=None, names=['id', 'employer'], index_col=None)
    existing_ids = set(emp_data['id'].values)

    remaining_id_file = os.path.join(p.parent_dir, 'data/loanstats/scraped_data/remaining_ids.txt')
    #remaining_id_file = os.path.join(p.parent_dir, 'data/loanstats/scraped_data/relisted_ids.txt')
    #remaining_id_file = os.path.join(p.parent_dir, 'data/loanstats/scraped_data/all_ids.txt')
    remaining_ids = set(np.loadtxt(remaining_id_file).astype(int))
    remaining_ids = remaining_ids.difference(existing_ids) 

    N = len(remaining_ids)
    print 'Need {} more datapoints'.format(N)

    if N > 0:
        accounts = list()
        for i in range(len(acct_names)):
            accounts.append(PausedAccount(acct_names[i%len(acct_names)]))
            sleep(5)
        accs = len(accounts)

        last_acc = accounts[0]
        with open(emp_data_file, 'a') as f:
            for i, id in enumerate(remaining_ids):
                if i>max_num or last_acc.auth_count >= max_fails:
                    break
                next_call, next_acc = min([(acc.next_call, acc) for acc in accounts])
                sleep_time = max(0, (next_acc.next_call - dt.now()).total_seconds())
                print acc_sleep_string(accounts, last_acc)
                sleep_func(sleep_time)
                company = next_acc.get_name(id)
                if company is None:
                    accounts.remove(next_acc)
                    if len(accounts) == 0:
                        break 
                    else:
                        continue
                else:
                    company = company.replace('|','/')
                last_acc = next_acc
                write_str = '{}|{}'.format(id, company)
                space_str = ' ' * max(0, 45 - len(write_str))
                print N-i, i, last_acc.auth_count, write_str[:45], space_str,
                f.write(write_str)
                f.write('\n')
                f.flush()

if __name__=='__main__':
    import sys
    print sys.argv
    if len(sys.argv)==3:
        max_fails = int(sys.argv[2])
    else:
        max_fails = 3
    max_tries = int(sys.argv[1])
    update(max_tries, max_fails)
