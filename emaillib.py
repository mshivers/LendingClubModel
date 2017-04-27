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

