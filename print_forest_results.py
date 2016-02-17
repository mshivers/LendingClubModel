printstr = '{:1.3f},\t {:1.2f},\t {:1.2f},\t {:1.2f}, \t{:1.0f}\n'
int_ranges = [[12,15],[15,20],[20,30]]
data_str = ''
for int_range in int_ranges:
    data_str += 'Int Range: [{},{}]\n'.format(*int_range)
    cdx = np.all(zip(x_test[:,1]>=int_range[0], x_test[:,1]<=int_range[1]), 1)
    data_str += 'Exp, \tAct, \tROE, \tint_rate, \tnum loans\n'
    pctls = np.arange(0,101,10)
    cutoffs = np.percentile(pf[cdx,1], pctls)
    for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
        idx = np.all(zip(pf[cdx,1] >= lower, pf[cdx,1]<upper), axis=1)
        default = 100*y_test[cdx][idx].mean()
        int_rate = x_test[cdx,1][idx].mean()
        roe = int_rate - default
        data = (100*np.mean([lower,upper]), default, roe, int_rate, sum(idx))
        data_str += printstr.format(*data)
print data_str


'''
titlestr = '{:>8s}'*7 + '\n'
printstr = '{:>8.2f}'*6 + '{:>8.0f}\n'
int_ranges = [[0,7],[7,10],[10,12],[12,13.5],[13.5,15],[15,17],[17,20],[20,30]]
for int_range in int_ranges:
    data_str += '\nInt Range: [{},{}]\n'.format(*int_range)
    data_str += titlestr.format('LAlpha','UAlpha','ROE','DExp','DAct','Rate','Num')
    cdx = np.all(zip(x_test[:,1]>=int_range[0], x_test[:,1]<=int_range[1]), 1)
    alphas = x_test[cdx,1] - 100*pf[cdx,1]
    pctls = np.arange(0,101,10)
    cutoffs = np.percentile(alphas, pctls)
    for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
        idx = np.all(zip(alphas>=lower, alphas<upper), axis=1)
        act_default = 100*y_test[cdx][idx].mean()
        exp_default =100*pf[cdx,1][idx].mean()
        int_rate = x_test[cdx,1][idx].mean()
        roe = int_rate - act_default
        data = (lower,upper, roe, exp_default,act_default, int_rate, sum(idx))
        data_str += printstr.format(*data)
print data_str

'''
titlestr = '{:>8s}'*7 + '\n'
printstr = '{:>8.2f}'*6 + '{:>8.0f}\n'
data_str = ''
data_str += titlestr.format('LAlpha','UAlpha','ROE','ExpD','ActD','Rate','Num')
alphas = x_test[:,1] - 100*pf[:,1]
pctls = np.arange(0,101,1)
cutoffs = np.percentile(alphas, pctls)
for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
    idx = np.all(zip(alphas >= lower, alphas<upper), axis=1)
    default = 100*y_test[idx].mean()
    int_rate = x_test[:,1][idx].mean()
    act_default = 100*y_test[idx].mean()
    exp_default =100*pf[:,1][idx].mean()
    roe = int_rate - act_default
    data = (lower,upper, roe, exp_default,act_default, int_rate, sum(idx))
    data_str += printstr.format(*data)
print data_str

'''
titlestr = '{:>8s}'*7 + '\n'
printstr = '{:>8.2f}'*6 + '{:>8.0f}\n'
int_ranges = [[0,7],[7,10],[10,12],[12,13.5],[13.5,15],[15,17],[17,20],[20,30]]
for int_range in int_ranges:
    data_str += '\nInt Range: [{},{}]\n'.format(*int_range)
    data_str += titlestr.format('LAlpha','UAlpha','ROE','DExp','DAct','Rate','Num')
    cdx = np.all(zip(x_test[:,1]>=int_range[0], x_test[:,1]<=int_range[1]), 1)
    alphas = x_test[cdx,1] - 100*pf[cdx,1]
    pctls = np.arange(0,101,10)
    cutoffs = np.percentile(alphas, pctls)
    for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
        idx = np.all(zip(alphas>=lower, alphas<upper), axis=1)
        act_default = 100*y_test[cdx][idx].mean()
        exp_default =100*pf[cdx,1][idx].mean()
        int_rate = x_test[cdx,1][idx].mean()
        roe = int_rate - act_default
        data = (lower,upper, roe, exp_default,act_default, int_rate, sum(idx))
        data_str += printstr.format(*data)
print data_str
'''
