# reads the monthly labor force size, and number of unemployed by fips county code,
# and constructs historical employment statistics by zip code for model fitting
import os
import json
import pandas as pd
import numpy as np

parent_dir = '/Users/marcshivers/LCModel'
z2f = json.load(file(os.path.join(parent_dir, 'zip3_fips.json'),'r'))

labor_force = pd.read_csv(os.path.join(parent_dir, 'labor_force.csv'), index_col=0)
labor_force = labor_force.fillna(0).astype(int).rename(columns=lambda x:int(x))
unemployed = pd.read_csv(os.path.join(parent_dir, 'unemployed.csv'), index_col=0)
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
    
