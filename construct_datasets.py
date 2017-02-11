import pandas as pd
import numpy as np
import json
import os
import requests
from collections import defaultdict, Counter


parent_dir = '/Users/marcshivers/LCModel/'
def build_zip3_to_location_names():
    import bls 
    cw = pd.read_csv('CBSA_FIPS_MSA_crosswalk.csv')
    grp = cw.groupby('FIPS')
    f2loc = dict([(k,list(df['CBSA Name'].values)) 
                  for k in grp.groups.keys()
                  for df in [grp.get_group(k)]])

    z3f = json.load(open('zip3_fips.json','r'))
    z2loc = dict()
    for z,fips in z3f.items():
        loc_set = set()
        for f in fips:
            if f in f2loc.keys():
                loc_set.update([bls.convert_unicode(loc) for loc in f2loc[f]])
        z2loc[int(z)] = sorted(list(loc_set))
     
    for z in range(1,1000):
        if z not in z2loc.keys() or len(z2loc[z])==0:
            z2loc[z] = ['No location info for {} zip'.format(z)]

    json.dump(z2loc, open('zip2location.json','w'))

def build_zip3_to_primary_city():
    data= pd.read_csv('zip_code_database.csv')
    data['place'] = ['{}, {}'.format(c,s) for c,s in zip(data['primary_city'].values, data['state'].values)]

    z2city = defaultdict(lambda :list())
    for z,c in zip(data['zip'].values, data['place'].values):
        z2city[int(z/100)].append(c)

    z2primarycity = dict()
    z2primary2 = dict()
    for z,citylist in z2city.items():
        z2primarycity[z] = Counter(citylist).most_common(1)[0][0]
        z2primary2[z] = Counter(citylist).most_common(2)

    for i in range(0,1000):
        if i not in z2primarycity.keys():
            z2primarycity[i] = 'No primary city for zip3 {}'.format(i)

    json.dump(z2primarycity, open('zip2primarycity.json','w'))


def save_charity_pct():
    irs = pd.read_csv('/Users/marcshivers/Downloads/12zpallagi.csv')
    irs['zip3'] = irs['zipcode'].apply(lambda x:int(x/100))
    irs = irs.ix[irs['AGI_STUB']<5]
    grp = irs.groupby('zip3')
    grp_sum = grp.sum()
    tax_df = pd.DataFrame({'agi':grp_sum['A00100'], 'charity':grp_sum['A19700']})
    tax_df['pct'] = tax_df['charity'] * 1.0 / tax_df['agi']
    json.dump(tax_df['pct'].to_dict(), open(os.path.join(parent_dir, 'charity_pct.json'), 'w'))



#def build_zip3_to_hpi():
z2c = pd.read_csv('zip2cbsa.csv')
z2c['zip3'] = z2c['ZIP'].apply(lambda x: int(x/100))
z2c = z2c[z2c['CBSA']<99999]
grp = z2c.groupby('zip3')

z2clist = dict()
for z3 in grp.groups.keys():
    g = grp.get_group(z3)
    z2clist[z3] = sorted(list(set(g['CBSA'].values)))


# get metro hpi for main areas
link = "http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_metro.csv"
cols = ['Location','CBSA', 'yr','qtr','index', 'stdev']
try:
    data = pd.read_csv(link, header=None, names=cols)
    data.to_csv(os.path.join(parent_dir, 'HPI_AT_metro.csv'))
except:
    print 'Failed to read FHFA website HPI data; using cached data'
    data = pd.read_csv(os.path.join(parent_dir,'HPI_AT_metro.csv'), header=None, names=cols)

data = data[data['index']!='-']
data['index'] = data['index'].astype(float)
data['yyyyqq'] = 100 * data['yr'] + data['qtr'] 
data = data[data['yyyyqq']>199000]

index = np.log(data.pivot('yyyyqq', 'CBSA', 'index'))
hpa1q = index - index.shift(1) 
hpa1y = index - index.shift(4) 
hpa5y = index - index.shift(20)
hpa10y = index - index.shift(40)

hpa1 = dict()
hpa4 = dict()
hpa20 = dict()
hpa40 = dict()
for z,c in z2clist.items():
    hpa1[z] = hpa1q.ix[:,c].mean(1)
    hpa4[z] = hpa1y.ix[:,c].mean(1)
    hpa20[z] = hpa5y.ix[:,c].mean(1)
    hpa40[z] = hpa10y.ix[:,c].mean(1)
hpa1 = pd.DataFrame(hpa1).dropna(axis=1, how='all')
hpa4 = pd.DataFrame(hpa4).dropna(axis=1, how='all')
hpa20 = pd.DataFrame(hpa20).dropna(axis=1, how='all')
hpa40 = pd.DataFrame(hpa40).dropna(axis=1, how='all')

hpa1.to_csv('hpa1.csv')
hpa4.to_csv('hpa4.csv')
hpa20.to_csv('hpa20.csv')
hpa40.to_csv('hpa40.csv')

'''
# get non-metro hpi, for other zip codes
link='http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_nonmetro.xls'
try:
    data = pd.read_excel(link, skiprows=2)
    data.to_csv(os.path.join(parent_dir, 'HPI_AT_nonmetro.csv'))
except:
    data = pd.read_csv(os.path.join(parent_dir,'HPI_AT_nonmetro.csv'))

grp = data.groupby('State')
tail5 = grp.tail(21).groupby('State')['Index']
chg5 = np.log(tail5.last()) - np.log(tail5.first())
tail1 = grp.tail(5).groupby('State')['Index']
chg1 = np.log(tail1.last()) - np.log(tail1.first())
chg = 100.0 * pd.DataFrame({'1yr':chg1, '5yr':chg5})

return chg
'''

# downloads the monthly non-seasonally adjusted employment data, and saves csv files for
# monthly labor force size, and number of unemployed by fips county code, to use to construct
# historical employment statistics by zip code for model fitting
z2f = json.load(file(os.path.join(parent_dir, 'zip3_fips.json'),'r'))
all_fips = []
[all_fips.extend(f) for f in z2f.values()]
fips_str = ['0'*(5-len(str(f))) + str(f) for f in all_fips]

data_code = dict()
data_code['03'] = 'unemployment_rate'
data_code['04'] = 'unemployment'
data_code['05'] = 'employment'
data_code['06'] = 'labor force'

#series_id = 'CN{}00000000{}'.format(fips, '06') 
cols = ['series_id', 'year', 'period', 'value']
link1 = 'http://download.bls.gov/pub/time.series/la/la.data.0.CurrentU10-14'
link2 = 'http://download.bls.gov/pub/time.series/la/la.data.0.CurrentU15-19'
cvt = dict([('series_id', lambda x: str(x).strip()) ])
data1 = pd.read_csv(link1, delimiter=r'\s+', usecols=cols, converters=cvt)
data2 = pd.read_csv(link2, delimiter=r'\s+', usecols=cols, converters=cvt)
data = pd.concat([data1, data2], ignore_index=True)
data = data.replace('-', np.nan)
data = data.dropna()
data = data.ix[data['period']!='M13']
data['value'] = data['value'].astype(float)
data['yyyymm'] = 100 * data['year'] + data['period'].apply(lambda x: int(x[1:]))
data['fips'] = [int(f[5:10]) for f in data['series_id']]
data['measure'] = [f[-2:] for f in data['series_id']]
data['region'] = [f[3:5] for f in data['series_id']]
del data['year'], data['period'], data['series_id']
county_data = data.ix[data['region']=='CN']
labor_force = county_data[county_data['measure']=='06'][['fips','yyyymm','value']]
labor_force = labor_force.pivot('yyyymm','fips','value')
unemployed = county_data[county_data['measure']=='04'][['fips','yyyymm','value']]
unemployed = unemployed.pivot('yyyymm','fips','value')
labor_force.to_csv(os.path.join(parent_dir, 'labor_force.csv'))
unemployed.to_csv(os.path.join(parent_dir, 'unemployed.csv'))

# reads the monthly labor force size, and number of unemployed by fips county code,
# and constructs historical employment statistics by zip code for model fitting
labor_force = labor_force.fillna(0).astype(int).rename(columns=lambda x:int(x))
unemployed = unemployed.fillna(0).astype(int).rename(columns=lambda x:int(x))

urates = dict()
for z,fips in z2f.items():
    ue = unemployed.ix[:,fips].sum(1)
    lf = labor_force.ix[:,fips].sum(1)
    ur = ue/lf
    ur[lf==0]=np.nan
    urates[z] = ur

urate = pd.DataFrame(urates)
urate.to_csv(os.path.join(parent_dir, 'urate_by_3zip.csv'))
    
