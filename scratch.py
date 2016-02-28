#Notes:
#    1. use 'Blank' instead of 'blank' so it doesn't get lumped in with lower case feature





# downloads the monthly non-seasonally adjusted employment data, and saves csv files for
# monthly labor force size, and number of unemployed by fips county code, to use to construct
# historical employment statistics by zip code for model fitting
import requests
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parent_dir = '/Users/marcshivers/LCModel'
default_curves = json.load(open(os.path.join(parent_dir, 'default_curves.json'), 'r'))
prepay_curves = json.load(open(os.path.join(parent_dir, 'prepay_curves.json'), 'r'))

'''

all_grades = list('ABCDEFG')
curves = prepay_curves
for term in [36,60]:
    plt.figure()
    for grade in all_grades:
        data = curves['{}{}'.format(grade,term)]
        #C = curves['D{}'.format(term)]
        #C = np.r_[C[0], np.diff(np.array(C))]
        #data = np.r_[data[0], np.diff(np.array(data))]  
        plt.plot(data)
    plt.title('Term: {}'.format(term))
    plt.legend(all_grades)
    plt.grid()
    plt.show()

'''
def make_loan(grade, term, rate, amount):
    pmt = np.pmt(rate/1200., term, amount)
    loan = dict([('grade', grade),('term', term),('monthly_payment', abs(pmt)),
        ('loan_amount', amount), ('int_rate', rate)])
    loan['default_max'] = default_curves['{}{}'.format(grade,term)][11]
    loan['prepay_max'] = 0.24
    return loan



def calc_npv(l, discount_rate=0.10):
    ''' All calculations assume a loan amount of $1.
    Note the default curves are the cumulative percent of loans that have defaulted prior 
    to month m; the prepayment curves are the average percentage of outstanding balance that is prepaid 
    each month (not cumulative) where the average is over all loans that have not yet matured (regardless 
    of prepayment or default).  We'll assume that the prepayments are in full'''

    net_payment_pct = 0.99  #LC charges 1% fee on all incoming payments

    key = '{}{}'.format(min('E', l['grade']), l['term']) 
    print key
    prepay_rate = np.array(prepay_curves[key])
    base_defaults = np.array(default_curves[key])
    
    risk_factor = 1.5
    cdefaults = (risk_factor * np.r_[base_defaults[:1],np.diff(base_defaults)]).cumsum()
    print cdefaults[11]
    #prepay_rate[:] = 0
    #cdefaults[:] = 0

    monthly_int_rate = l['int_rate']/1200.
    monthly_discount_rate = (1 + discount_rate) ** (1/12.) - 1
    monthly_payment = l['monthly_payment'] / l['loan_amount']

    # start with placeholder for time=0 investment for irr calc later
    payments = np.zeros(l['term']+1)
    
    principal_balance = 1
    # add monthly payments
    for m in range(1, l['term']+1):

        interest_due = principal_balance * monthly_int_rate
        principal_due = monthly_payment - interest_due

        # prepayment rate is a pct of ending balance
        principal_balance -= principal_due
        prepayment_amt = principal_balance * prepay_rate[m-1]
        
        scheduled_amt = monthly_payment * (1 - cdefaults[m-1])
        payments[m] = prepayment_amt + scheduled_amt
        
        # reduce monthly payment to reflect this month's prepayment
        principal_balance -= prepayment_amt
        monthly_payment *= (1 - prepay_rate[m-1])


    # reduce payments by lending club service charge
    payments *= net_payment_pct
    npv = np.npv(monthly_discount_rate, payments) 

    # Add initial investment outflow at time=0 to calculate irr: 
    payments[0] += -1
    irr = np.irr(payments)
    
    l['irr'] = -1 + (1 + irr) ** 12
    l['npv'] = 100 * npv    

    return l, base_defaults, cdefaults 
    



'''
hpa4 = pd.read_csv(os.path.join(parent_dir, 'hpa4.csv'), index_col = 0)







fld = 'clean_title'
grp = df[['wgt_default', fld]].groupby(fld)
d = [i for i in zip(grp.count().index, grp.count().values.squeeze(), grp.mean().values.squeeze()) if i[1]>50]
d = sorted(d, key=lambda x:x[1])




ldx = df['emp_title'].apply(lambda x:str(x)[-1].islower())
udx = df['emp_title'].apply(lambda x:np.all([c.isupper() for c in str(x)]))
out = list()
for tfw,_,_ in d:
    idx = (df['title_first_word']==tfw) 
    lower = df[ldx & idx][iv].mean()
    upper = df[udx & idx][iv].mean()
     
    if sum(ldx&idx) > 50 and sum(udx&idx) > 50:
        out.append((tfw, lower, upper))
        print '{:>20s} {:1.1f}%, {:1.1f}%, {:1.2f}%'.format(tfw+':', 100*lower, 100*upper, 100*(lower-upper))

lu = pd.DataFrame(out, columns=['ftw', 'lower','upper'])

c = Counter()

N=8
for w in df['clean_title'].values:
    w = ' '*max(N-len(w),0) + w
    c.update([w[i:i+N] for i in range(len(w)-N+1)])
c.most_common(100)


c = Counter()
for w in df['clean_title'].values:
    s = w.split()
    if len(s) > 2:
        c.update(s[1:-1])













def calc_default_loss():
    T = 36.

    c = Counter(y10b.ix[y10b['term']==T,'round_mths'].values)
    tmp = sorted(list(c.items()))
    xx = [t[0] for t in tmp]
    yy = [t[1] for t in tmp]
    plt.plot(xx,yy)
    xx = np.array(xx)
    yy = np.array(yy)
    rl = T - xx
    rf = rl / T
    dprob = 1.0*yy / sum(yy)
    loss = rf.dot(dprob)
    mult = loss / sum(dprob[:3])
    plt.plot(xx,yy)
    print dprob, loss, mult

#employer name was in the emp_title field before 9/24/13
fld_name = 'clean_title'
tok_len = 4
idx = (df['issue_d']>=np.datetime64('2013-10-01')) & (df['issue_d']<=np.datetime64('2014-05-01'))
training = df[idx].copy()

toks = list()
titles = training[fld_name].apply(lambda x:'^{}$'.format(x)) #add string boundary tokens

for ct in titles.values:
    toks.extend([ct[i:i+tok_len] for i in range(max(1,len(ct)-tok_len+1))])

tok_df = pd.DataFrame(Counter(toks).items(), columns=['tok', 'freq'])
tok_df = tok_df.sort('freq',ascending=False)

odds_map = dict()
mean_default = training['12m_wgt_default'].mean() 
for _, row in tok_df.iterrows():
 
    tok, freq = row
    if freq<2000:
        continue
    training['has_tok'] = titles.apply(lambda x: tok in x)
    grp = training.groupby('has_tok')
    default_sum = grp.sum()['12m_wgt_default']
    default_count = grp.count()['12m_wgt_default']
    default = default_sum  / default_count 
    log_odds = np.log(default[True]) - np.log(default[False])
    print default_count[True], '"{}"'.format(tok), '{:1.2f}'.format(log_odds)
    print default
    print '\n'
    odds_map[tok] = (default_sum[True], default_count[True], default_count[False]) 

json.dump(odds_map, open(os.path.join(parent_dir, fname),'w'))

'''
