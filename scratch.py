#Notes:
#    1. use 'Blank' instead of 'blank' so it doesn't get lumped in with lower case feature



'''
all_grades = list('ABCDEFG')

out = list()
selected_notes = inotes #[n for n in inotes if n['issueDate'] is not None and not n['issueDate'][:4]=='2015']
for term in [36,60]:
    for grade in all_grades:
        grade_notes = [n for n in selected_notes if n['grade'].startswith(grade) and n['loanLength']==term]
        if len(grade_notes)>10:
            tmp = pa.calc_monthly_returns(grade_notes)
            cash = (tmp.total_interest - tmp.expected_defaults).sum()
            invested = 0.5*(tmp.total_invested + tmp.current_balance).sum()
            current = tmp.current_balance.sum()
            avg_age = (tmp.total_invested * tmp.age).sum() / tmp.total_invested.sum()
            out.append([ grade, term,avg_age, cash, invested,current, (1 + cash/invested)**(1 / avg_age)])
out = pd.DataFrame(out, columns=['grade', 'term', 'avg_age', 'cash', 'invested','current', 'return'])
print out

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


wynd = list()
ids = set()
for i, l in enumerate(ira):
    print i
    idx = stats['id']==l['loanId']
    if idx.sum():
        zip = stats.ix[idx, 'zip_code'].values[0]
        zip = str(int(zip))
        if str(zip) in z2loc.keys():
            for loc in z2loc[zip]:
                loc = str(loc)
                if isinstance(loc, str):
                    if loc.strip().endswith('NM') or loc.strip().endswith('OK'):
                        if l['loanId'] not in ids:
                            wynd.append(l)
                            ids.add(l['loanId'])

                        print l, '\n\n\n'

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

def make_loan(grade, term, rate, amount):
    pmt = np.pmt(rate/1200., term, amount)
    loan = dict([('grade', grade),('term', term),('monthly_payment', abs(pmt)),
        ('loan_amount', amount), ('int_rate', rate)])
    loan['default_max'] = default_curves['{}{}'.format(grade,term)][11]
    loan['prepay_max'] = 0.24
    return loan



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
