from datetime import datetime as dt, timedelta as td
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import itertools as it
from collections import Counter, defaultdict
import os
import json
import loanstats

def substrings(x):
    toks = list()
    x = '^{}$'.format(x)
    #add all whole words
    toks.extend(x.split())
    toks.extend([x[i:i+k] for k in range(3,min(7, len(x)+1)) 
        for i in range(max(1,len(x)-k+1))])
    return list(set(toks))

#construct X matrix
M = len(tok_dx)
N_train = len(is_df)
X_train = sp.sparse.dok_matrix((N_train,M))
for row, ct in enumerate(is_df['empTitle'].values):
    print(row, N_train, ct)
    toks = substrings(ct) 
    for tok in toks:
        if tok in tok_dx.keys():
            X_train[row,tok_dx[tok]] = 1

1/0


if 'df' not in locals().keys():
    df = loanstats.load_training_data()
    isdx = df.in_sample
    oos = ~isdx 
    
    is_df = df.ix[isdx,['empTitle', '12m_wgt_default', 'subGrade']].copy()
    oos_df = df.ix[oos,['empTitle', '12m_wgt_default', 'subGrade']].copy()

    tok_count = Counter()
    for i, ct in enumerate(is_df['empTitle'].values):
        tok_count.update(substrings(ct))
        if i%1000==0:
            print(i, len(tok_count))

    tok_df = pd.DataFrame(tok_count.most_common(), columns=['tok', 'freq'])
    tok_df = tok_df.sort('freq',ascending=False)


    #construct index value for each token in input array
    vocab_sz=sum(tok_df['freq']>=100)
    tok_dx = dict(zip(tok_df['tok'].values[:vocab_sz], range(vocab_sz)))

    #construct X matrix
    M = len(tok_dx)
    N_train = len(is_df)
    X_train = sp.sparse.dok_matrix((N_train,M))
    for row, ct in enumerate(is_df['empTitle'].values):
        print(row, N_train, ct)
        toks = substrings(ct) 
        for tok in toks:
            if tok in tok_dx.keys():
                X_train[row,tok_dx[tok]] = 1

    N_test = len(oos_df)
    X_test = sp.sparse.dok_matrix((N_test,M))
    for row, ct in enumerate(oos_df['clean_title'].values):
        print(row, N_test, ct)
        toks = substrings(ct) 
        for tok in toks:
            if tok in tok_dx.keys():
                X_test[row,tok_dx[tok]] = 1

    y_train = is_df['12m_wgt_default'] > 0.5
    y_test = oos_df['12m_wgt_default'] > 0.5

    num2tok = dict(zip(tok_dx.values(), tok_dx.keys()))

    hy_test = oos_df['subGrade']>'C6'
    hy_train = is_df['subGrade']>'C6'

    idx = np.array([k for (k,v) in num2tok.items() if len(v)==4 or  
        (len(v)<=6 and (v.startswith('^') or v.startswith(' ')) and (v.endswith('$') or v.endswith(' ')))])


    X_subtrain = X_train[:,idx]
    X_subtest = X_test[:,idx]
    y_subtrain = y_train
    y_subtest = y_test

max_x = X_subtrain.shape[1]
C = 0.15  
clf = LogisticRegression(C=C, penalty ='l1', tol=0.0001)
clf.fit(X_subtrain,y_subtrain)

prob_default = clf.predict_proba(X_subtest)[:,1]
c_tmp = Counter(prob_default)
pctls = np.percentile(prob_default, range(20,91,20))
tmp = pd.DataFrame(y_subtest)
tmp['prob'] = prob_default
bins = np.unique([-1]+list(pctls) + [101])
g = tmp.groupby(np.digitize(prob_default, bins))
means = g.mean()['12m_wgt_default']
low = pctls[0]
high = pctls[-1]
low_pct = y_subtest[prob_default<=low].mean()
high_pct = y_subtest[prob_default>=high].mean()

num_nonzeros = (clf.coef_!=0).sum()
pct_nonzero = 100*(clf.coef_!=0).mean()
print( 'C={:1.2f}: {} of {} nonzero or {:1.3f}% -- {:1.2f}% -- '.format(C, 
        num_nonzeros, max_x, pct_nonzero, 100*(high_pct-low_pct)), )
for m in means:
    print('{:1.2f}% / '.format(100*m),)
bads = 100*prob_default[np.where(y_subtest==True)[0]].mean() 
goods = 100*prob_default[np.where(y_subtest==False)[0]].mean()
print('{:1.2f}% vs {:1.2f}% ({:1.2f}%) / '.format(bads, goods, bads-goods),)
      
print(c_tmp.most_common(1)[0][1])

ix=np.where(clf.coef_[0])[0]
coef= clf.coef_[0][ix]
coefs = list()
save_params = list()
for a,b in zip(ix, coef):
    save_params.append((num2tok[idx[a]],b))
    coefs.append((b,a,num2tok[idx[a]]))
coefs = sorted(coefs)
save_dict = dict(save_params)
json.dump(save_dict, open('logistic_regression_clean_title_log_odds_v2.json','w'))
    
