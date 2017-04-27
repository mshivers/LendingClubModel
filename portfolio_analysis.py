from datetime import datetime as dt, timedelta as td
from time import sleep
import numpy as np
import pandas as pd
import os
import lclib
import json
import requests
from personalized import p
from constants import DefaultProb
from collections import Counter

default_wgt = lambda n :DefaultProb.by_status(n['loanStatus'])

class PortfolioAnalysis(object):
    def __init__(self, notes=None, account=None):
        self.account = account
        if notes==None:
            self.notes = self.get_notes(account)
        else:
            self.notes = notes

    def _convert_to_date(self, dtstr):
        try:
            return dt.strptime(dtstr.split('T')[0], '%Y-%m-%d')
        except:
            return dtstr

    def get_notes(self, account=None):
        if account == None:
            account = 'all'
            notes = self.get_notes('ira') + self.get_notes('tax')
        else:
            id = p.get_id(account)
            key = p.get_key(account)

            notes = []
            url = 'https://api.lendingclub.com/api/investor/v1/accounts/{}/detailednotes'.format(id)
            result = requests.get(url, headers={'Authorization': key})
            if result.status_code == 200:  #success
                result_js = result.json()
                if 'myNotes' in result_js.keys():
                    notes = result.json()['myNotes']
            for note in notes:
                note['account'] = account
                for k,v in note.items():
                    if k.endswith('Date'):
                        note[k+'time'] = self._convert_to_date(v)
        self.notes = notes
        self.account = account
        return notes

    def get_grade(self, grade):
        if len(grade)==1:
            notes = [note for note in self.notes if note['grade'][:1] == grade]
        else:
            notes = [note for note in self.notes if note['grade'] == grade]
        return notes

    def notes_by_status(self, status):
        return [note for note in self.notes if note['currentPaymentStatus']==status]

    def days_since_payment(self, note):
        if note['lastPaymentDatetime'] is not None:
            return (dt.now() - note['lastPaymentDatetime']).days 
        else:
            return (dt.now() - note['issueDatetime']).days 

    def status_by_issue_month(self):
        df = pd.DataFrame(self.notes)
        df['firstDatetime'] = df['issueDatetime']
        idx = df.firstDatetime.isnull()
        df.ix[idx, 'firstDatetime'] = df.ix[idx, 'orderDatetime']
        df['issueMth'] = df['firstDatetime'].apply(lambda x:x.strftime('%Y%m'))
        by_status = df.groupby(['issueMth', 'loanStatus'])['principalPending'].sum().unstack()
        invested = df.groupby('issueMth')['principalPending'].sum()
        by_status = by_status.div(invested, axis=0)
        by_status['invested'] = invested
        return by_status


def parse_date(dtstr):
    return dt.strptime(dtstr.split('T')[0], '%Y-%m-%d')


def get_seasoned_loans(loans):
    seasoned = [l for l in loans if l['loanStatus'].startswith('Current')]
    for l in seasoned:
        start = parse_date(l['issueDate'])
        l['dob'] = (dt.now() - start).days 

    seasoned = [l for l in seasoned if l['dob']>365]
    seasoned = sorted(seasoned, key=lambda x: x['dob'])

    df = pd.concat([pd.Series(seasoned[i]) for i in range(len(seasoned))], axis=1).T
    return df[['loanId', 'noteId', 'noteAmount', 'loanStatus', 'dob', 'principalPending', 
        'grade', 'interestRate','loanLength', 'paymentsReceived', 'creditTrend']]


def get_late_loans(loans):
    late = [l for l in loans if l['loanStatus'].startswith('Late (31')]
    for l in late:
        if l['lastPaymentDate']:
            start = parse_date(l['lastPaymentDate'])
        else:
            start = parse_date(l['issueDate'])
        l['late_days'] = (dt.now() - start).days
    late = sorted(late, key=lambda x: x['late_days'])
    df = pd.concat([pd.Series(late[i]) for i in range(len(late))], axis=1).T
    return df[['loanId', 'noteId', 'noteAmount', 'loanStatus', 'currentPaymentStatus', 'late_days',
        'principalPending', 'grade', 'interestRate','loanLength', 'paymentsReceived', 'creditTrend']]


def monthly_payment(loan, rate, term):
    i = float(rate)/100/12
    return loan * i * (1+i)**term / ((1+i)**term - 1)


def cumulative_scheduled_interest(loan, rate, term, num_periods):
    num_periods = min(num_periods, term)
    interest_paid = 0
    principal_paid = 0
    remaining_principal = loan
    monthly_rate = float(rate)/100/12
    payment = monthly_payment(loan, rate, term)
    for i in range(num_periods):
        interest_payment = remaining_principal * monthly_rate
        principal_payment = payment - interest_payment 
        interest_paid += interest_payment 
        principal_paid += principal_payment 
        remaining_principal -= principal_payment
    return interest_paid, remaining_principal


def total_interest_received(notes, num_months=None):
    if num_months == None:
        return sum([n['interestReceived'] for n in notes])
    else:
        total_interest = 0
        for n in notes:
            exp_interest = cumulative_scheduled_interest(n['noteAmount'],
                    n['interestRate'], n['loanLength'], num_months)[0]
            total_interest += min(n['interestReceived'],exp_interest)
        return total_interest

def expected_write_offs(notes, num_months=None):
    if num_months == None:
        return sum([default_wgt(n) * n['principalPending'] for n in notes if n['issueDate'] is not None])
    else:
        exp_write_off = 0
        for n in notes:
            wgt = default_wgt(n)
            if 'lastPaymentDate' not in n.keys() or n['lastPaymentDate'] is None:
                exp_write_off += wgt * n['principalPending']
            else:
                issue_dt = parse_date(n['issueDate'])
                last_payment_date = parse_date(n['lastPaymentDate'])
                if (last_payment_date - issue_dt).days < num_months * 30 + 15:
                    exp_write_off += wgt * n['principalPending']
        return exp_write_off

        
def total_written_off(notes, num_months=None):
    if num_months == None:
        return sum([n['principalPending'] for n in notes if n['loanStatus']=='Charged Off'])
    else:
        total_write_off = 0
        for n in notes:
            if n['loanStatus'] == 'Charged Off':
                if 'lastPaymentDate' not in n.keys() or n['lastPaymentDate'] is None:
                    total_write_off += n['principalPending']
                else:
                    issue_dt = parse_date(n['issueDate'])
                    last_payment_date = parse_date(n['lastPaymentDate'])
                    if (last_payment_date - issue_dt).days < num_months * 30 + 15:
                        total_write_off += n['principalPending']
        return total_write_off

        
def total_invested(notes):
    return sum([n['noteAmount'] for n in notes if n['issueDate'] is not None])

def current_balance(notes):
    return sum([n['principalPending'] for n in notes if n['issueDate'] is not None])

def issue_months(notes):
    return sorted(list(set([n['issueDate'][:7] for n in notes if n['issueDate'] is not None])))


def calc_monthly_returns(notes):
    now = dt.now()
    out = list()
    for m in issue_months(notes):
        yr, mth = m.split('-')
        issue_dt = dt(int(yr), int(mth), 15)
        mth_notes = [n for n in notes if n['issueDate'] is not None 
                                     and n['issueDate'].startswith(m)]
        p = total_invested(mth_notes)
        c = current_balance(mth_notes)
        i = total_interest_received(mth_notes)
        d = total_written_off(mth_notes)
        e = expected_write_offs(mth_notes)
        avg_balance = (p + c)/2.0 

        invest_days = (now - issue_dt).days
        invest_years = 1.0 * invest_days / 365
        total_return = (1.0*i - e) / avg_balance 
        annual_return = ((1 + total_return) ** (1/invest_years)) - 1
        #print m, issue_dt, invest_years, total_return, annual_return, p,c, i, d, e
        out.append([ m, issue_dt, invest_years, total_return, annual_return, p,c, i, d, e])
        #print '{}: {}, {}, {}, {}'.format(m, annual_return, p, i, d)
    col_names = ['month', 'issue_dt', 'age', 'total_exp_return', 'annual_exp_return', 'total_invested', 
            'current_balance', 'total_interest', 'total_defaulted', 'expected_defaults']
    return pd.DataFrame(out, columns=col_names).set_index('month')

def initial_returns(notes, num_months):
    now = dt.now()
    out = list()
    for m in issue_months(notes):
        yr, mth = m.split('-')
        issue_dt = dt(int(yr), int(mth), 15)
        mth_notes = [n for n in notes if n['issueDate'] is not None 
                                     and n['issueDate'].startswith(m)]
        p = total_invested(mth_notes)
        c = current_balance(mth_notes)
        i = total_interest_received(mth_notes, num_months)
        d = total_written_off(mth_notes, num_months)
        e = expected_write_offs(mth_notes, num_months)
        avg_balance = (p + c)/2.0 

        invest_days = (now - issue_dt).days
        invest_years = 1.0 * invest_days / 365
        total_return = (1.0*i - d) / avg_balance 
        annual_return = ((1 + total_return) ** (1/invest_years)) - 1
        holding_mths = (now - issue_dt).days / (365.25/12)

        if num_months is not None:
            holding_mths = min(num_months, holding_mths)
        annualization = 12.0 / holding_mths
        exp_ret = (1+ (i-e)/avg_balance)**(annualization) - 1
        act_ret = (1+ (i-d)/avg_balance)**(annualization) - 1
        out.append((m, holding_mths, p,c, int(i), int(d), int(e), act_ret, exp_ret))
    cols = ['month', 'num_mths', 'invested', 'outstanding', 'interest', 'defaults', 
            'exp_defaults', 'act_return', 'exp_return']
    df = pd.DataFrame(data=out, columns=cols) 
    return df 


