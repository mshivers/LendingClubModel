import os
import json
import urllib
import requests
import smtplib
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
import utils
from personalized import p



email_keys = ['accOpenPast24Mths','mthsSinceLastDelinq', 'mthsSinceRecentBc', 'bcUtil', 'totCollAmt', 
        'isIncV', 'numTlOpPast12m', 'totalRevHiLim', 'mthsSinceRecentRevolDelinq', 'revolBal',
        'pubRec', 'delinq2Yrs', 'inqLast6Mths', 'numOpRevTl', 'pubRecBankruptcies', 'numActvRevTl',
        'mthsSinceRecentBcDlq', 'revolUtil', 'numIlTl', 'numRevTlBalGt0', 'numTl90gDpd24m', 'expDefaultRate', 
        'initialListStatus', 'moSinOldIlAcct', 'numBcTl', 'totHiCredLim', 'delinqAmnt', 'moSinOldRevTlOp', 
        'numRevAccts', 'totalAcc', 'mortAcc', 'mthsSinceRecentInq', 'moSinRcntRevTlOp','totCurBal', 
        'collections12MthsExMed', 'dti', 'numActvBcTl', 'pctTlNvrDlq', 'totalBcLimit',
        'accNowDelinq', 'numTl30dpd', 'percentBcGt75', 'numBcSats', 'openAcc', 'numAcctsEver120Ppd', 'bcOpenToBuy',
        'numTl120dpd2m', 'taxLiens', 'mthsSinceLastRecord', 'totalBalExMort', 'avgCurBal', 'moSinRcntTl', 
        'mthsSinceLastMajorDerog', 'totalIlHighCreditLimit', 'chargeoffWithin12Mths', 'clean_title_rank']

email_keys = [ 'isIncV', 'totalRevHiLim', 'revolBal', 'numRevTlBalGt0', 'numTl90gDpd24m',
        'initialListStatus', 'totHiCredLim', 'delinqAmnt', 'mortAcc', 'totCurBal', 
        'pctTlNvrDlq', 'totalBcLimit', 'numTl30dpd', 'percentBcGt75', 'numAcctsEver120Ppd',
        'numTl120dpd2m', 'totalBalExMort', 'avgCurBal',
        'totalIlHighCreditLimit', 'clean_title_rank']

def all_detail_str(loan):
    all_details = '\n'.join(sorted(['{}: {}'.format(k,v) for k,v in loan.items() 
        if k in email_keys or k.startswith('dflt')]))
    return all_details 


def send_email(msg):
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    email = p.get_email('smtp')
    server.login(email, p.get_key('smtp'))
    msg_list = ['From: {}'.format(email), 
                'To: {}'.format(email), 
                'Subject: Lending Club', '']
    msg_list.append(msg)
    msg = '\r\n'.join(msg_list) 
    server.sendmail(email, [email], msg)
    return


def email_details(acct, loans, info):
    msg_list = list()
    msg_list.append('{} orders have been staged to {}'.format(len(loans), acct))
    msg_list.append('{} total loans found, valued at ${:1,.0f}'.format(info['num_new_loans'], info['value_new_loans']))

    count_by_grade = dict(zip('ABCDEFG', np.zeros(7)))
    for loan in loans:
        if loan['email_details'] == True:
            count_by_grade[loan['grade']] += 1

    g = info['irr_df'].groupby(['grade', 'initialListStatus'])
    def compute_metrics(x):
        result = {'irr_count': x['base_irr'].count(), 'irr_mean': x['base_irr'].mean()}
        return pd.Series(result, name='metrics')
    msg_list.append(g.apply(compute_metrics).to_string())

    irr_msgs = list()
    irr_msgs.append('Average IRR is {:1.2f}%.'.format(100*info['average_irr']))
    for grade in sorted(info['irr_by_grade'].keys()):
        avg = 100*np.mean(info['irr_by_grade'][grade])
        num = len(info['irr_by_grade'][grade])
        bought = int(count_by_grade[grade])
        irr_msgs.append('Average of {} grade {} IRRs is {:1.2f}%; {} staged.'.format(num, grade, avg, bought))
    msg_list.append('\r\n'.join(irr_msgs))

    for loan in loans:
        if loan['email_details'] == True:
            msg_list.append(detail_str(loan))
            msg_list.append(all_detail_str(loan))
        loan['email_details'] = False
    msg_list.append('https://www.lendingclub.com/account/gotoLogin.action')
    msg_list.append('Send at MacOSX clocktime {}'.format(dt.now()))
    msg = '\r\n\n'.join(msg_list) 
    send_email(msg)
    return

