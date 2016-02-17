# downloads the monthly non-seasonally adjusted employment data, and saves csv files for
# monthly labor force size, and number of unemployed by fips county code, to use to construct
# historical employment statistics by zip code for model fitting
import requests
import json
import os
import pandas as pd
import numpy as np

parent_dir = '/Users/marcshivers/LCModel'
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


