## Lending Club's prepayment curves are all percentages of face value by month, 
## rather than percentage of outstanding face value, which we do.  Should we change?
import os
import json
import numpy as np
import scipy
from matplotlib import pyplot as plt
import loanstats
from constants import PathManager as paths
import pandas as pd
from personalized import p  

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from collections import defaultdict, Counter
from personalized import p
from constants import PathManager
from datalib import ReferenceData

''' This loads the monthly employment data for the trailing 14 months '''
print 'Downloading BLS data from bls.gov'
#link = 'http://www.bls.gov/lau/laucntycur14.txt'
link = 'http://www.bls.gov/web/metro/laucntycur14.txt'
cols = ['Code', 'StateFIPS', 'CountyFIPS', 'County', 
    'Period', 'CLF', 'Employed', 'Unemployed', 'Rate']
file = requests.get(link)
rows = [l.split('|') for l in file.text.split('\r\n') if l.startswith(' CN')]
data =pd.DataFrame(rows, columns=cols)
data['Period'] = data['Period'].apply(lambda x:dt.strptime(x.strip()[:6],'%b-%y'))

# keep only most recent 12 months; note np.unique also sorts
min_date = np.unique(data['Period'])[1]
data = data[data['Period']>=min_date]

# reduce Code to just state/county fips number
data['FIPS'] = data['Code'].apply(lambda x: int(x.strip()[2:7]))

# convert numerical data to floats
to_float = lambda x: float(str(x).replace(',',''))
for col in ['CLF', 'Unemployed']:
    data[col] = data[col].apply(to_float)
data = data.ix[:,['Period','FIPS','CLF','Unemployed']]
labor_force = data.pivot('Period', 'FIPS','CLF')
unemployed = data.pivot('Period', 'FIPS', 'Unemployed')

avg_urate_ttm = dict()
urate= dict()
urate_chg = dict()
z2f = ReferenceData().get_zip3_to_fips()
for z, fips in z2f.items():
    avg_unemployed = unemployed.ix[1:,fips].sum(1).sum(0) 
    avg_labor_force = labor_force.ix[1:,fips].sum(1).sum(0)
    avg_urate_ttm[z] = avg_unemployed / avg_labor_force 
    urate[z] =  unemployed.ix[-1,fips].sum(0) / labor_force.ix[-1,fips].sum(0)
    last_year_ur =  unemployed.ix[1,fips].sum(0) / labor_force.ix[1,fips].sum(0)
    urate_chg[z] = urate[z] - last_year_ur

summary = pd.DataFrame({'avg':pd.Series(avg_urate_ttm),
                        'current':pd.Series(urate),
                        'chg12m':pd.Series(urate_chg)})
summary.index.name = 'zip3'



'''
allpmts = open('data/loanstats/PMTHIST_ALL_20170315.csv', 'r'). readline().strip().upper().split(',')
invpmts = open('data/loanstats/PMTHIST_INVESTORS_20170417.csv', 'r'). readline().strip().upper().split(',')

inota = sorted([f for f in invpmts if f not in allpmts])
anoti = sorted([f for f in allpmts if f not in invpmts])

emp_data_file = os.path.join(p.parent_dir, 'data/loanstats/scraped_data/combined_data.txt')
comb_data = pd.read_csv(emp_data_file, sep='|', header=0, index_col=None)
comb_ids = set(comb_data['id'].values)

scrape_data_file = os.path.join(p.parent_dir, 'data/loanstats/scraped_data/SCRAPE_FILE.txt')
scrape_data = pd.read_csv(scrape_data_file, sep='|', header=0, index_col=None)
scrape_ids = set(scrape_data['id'].values)

remaining_id_file = os.path.join(p.parent_dir, 'data/loanstats/scraped_data/remaining_ids.txt')
remaining_ids = set([int(r) for r in open(remaining_id_file, 'r').read().split('\n')])

N = len(remaining_ids)
print 'Need {} more datapoints'.format(N)


#test_data = fit_data.ix[~fit_data.in_sample] 
#predictions = [tree.predict(x_test) for tree in forest.estimators_]
#predictions = np.vstack(predictions).T  #loans X trees
N = len(forest.estimators_)
score_data = list()
incr = 50
for num_trees in range(incr, N+1, incr):

    test_data['default_prob'] = np.percentile(predictions[:, :num_trees], 64, axis=1)
      
    res_data = list()
    grp = test_data.groupby(['grade', 'term'])
    default_fld = 'default_prob'
    for k in sorted(grp.groups.keys(), key=lambda x:(x[1], x[0])):
        sample = grp.get_group(k)
        grp_predict = sample[default_fld]
        pctl10, grp_median, pctl90 = np.percentile(grp_predict.values, [10,50,90])
        bottom = grp_predict<=pctl10
        top = grp_predict>=pctl90
        bottom_default_mean = 100*sample.ix[bottom, iv].mean()
        bottom_predict_mean = 100*grp_predict[bottom].mean()
        top_default_mean = 100*sample.ix[top, iv].mean() 
        top_predict_mean = 100*grp_predict[top].mean()
        rate_diff = sample.ix[bottom, 'intRate'].mean() - sample.ix[top, 'intRate'].mean()
        res_data.append([k, len(sample), bottom_default_mean, top_default_mean, 
                         bottom_predict_mean, top_predict_mean, rate_diff])
    cols = ['group', 'NObs', 'decile1_actual', 'decile10_actual', 'decile1_predicted', 'decile10_predicted', 
            'rate_diff']
    res = pd.DataFrame(res_data, columns=cols)
    res['decile1_error'] = res['decile1_predicted'] - res['decile1_actual']
    score = (res['NObs'] * res['decile1_error']).sum() / res['NObs'].sum()
    abs_score = (res['NObs'] * res['decile1_error']).abs().sum() / res['NObs'].sum()
    score_data.append((num_trees, score, abs_score))
    print '{}, {:1.2f}, {:1.2f}'.format(num_trees, score, abs_score)
    print res
    print '\n\n\n'
scores = pd.DataFrame(score_data, columns=['num_trees', 'score', 'score_abs']).set_index('num_trees')
plt.figure()
scores['score_abs'].plot()
plt.ylim(0,plt.ylim()[1])
plt.show()

for k1,v1 in api_mid.items():
    for k2,v2 in hist_mid.items():
        try: 
            if str(v1)==str(v2):
                print k1, k2
            elif float(v1)==float(v2):
                print k1, k2
        except:
            pass


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
