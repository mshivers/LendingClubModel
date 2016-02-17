from datetime import datetime as dt, timedelta as td
from time import sleep
from pprint import pprint
import pandas as pd
import numpy as np
import requests
import json
import os
from collections import defaultdict


parent_dir = '/Users/marcshivers/LCModel'

def _parse_urate_cn(name):
    name = name.upper().strip()
    name = name.replace('ST.', 'SAINT')
    name = name.replace(' COUNTY,', ',')
    name = name.replace(' PARISH,', ',')
    name = name.replace(' MUNICIPIO,', ',')
    name = name.replace(' CITY,', ',')
    name = name.replace(' BOROUGH,', ',')
    name = name.replace(' CENSUS AREA,', ',')
    name = name.replace(' MUNICIPALITY,', ',')
    name = name.replace(' BOROUGH/CITY,', ',')
    name = name.replace(' BOROUGH/MUNICIPALITY,', ',')
    name = name.replace(' COUNTY/CITY,', ',')
    name = name.replace(' COUNTY/TOWN,', ',')
    name = name.replace(' DISTRICT OF COLUMBIA,', 'DISTRICT OF COLUMBIA')
    return name.strip()

def _parse_census(name):
    name = name.upper()
    name = name.replace('ST.', 'SAINT')
    name = name.replace('(CITY)', '')
    return name.strip()


def construct_loc2cn():
    #Get city - county mapping
    loc2cn = defaultdict(lambda :dict())
    fdir = '/Users/marcshivers/Downloads/AllStatesFedCodes_20140802'
    fnames = os.listdir(fdir)
    for file in fnames:
        data=pd.read_csv(os.path.join(fdir, file), sep='|')
        if len(data)>0:
            state_data = data.groupby('STATE_ALPHA')
            for state in state_data.groups.keys():
                vals = state_data.get_group(state)
                county = vals['COUNTY_NAME'].apply(_parse_census)
                city = vals['FEATURE_NAME'].apply(_parse_census)
                pairs = zip(city, county)
                loc2cn[state].update(pairs)
    return loc2cn

def load_loc2cn():
    return json.load(open(os.path.join(parent_dir, 'loc2cn.json'), 'r'))
     
def convert_unicode(cn):
    if type(cn)==unicode:
        cn = cn.replace(u'\xf1', u'N')
        cn = cn.replace(u'\xa7', u'a')
        cn = cn.replace(u'\u0100', u'A')
        cn = cn.replace(u'\u0101', u'A')
        cn = cn.replace(u'\u0113', u'E')
        cn = cn.replace(u'\u012b', u'I')
        cn = cn.replace(u'\u014c', u'O')
        cn = cn.replace(u'\u014d', u'O')
        cn = cn.replace(u'\u016b', u'U')
        cn = cn.replace(u'\u2018', u'')
    elif type(cn)==str:
        cn = cn.replace('\xf1', 'N')
        cn = cn.replace('\xa7', 'a')
        cn = cn.replace('\u0100', 'A')
        cn = cn.replace('\u0101', 'A')
        cn = cn.replace('\u0113', 'E')
        cn = cn.replace('\u012b', 'I')
        cn = cn.replace('\u014c', 'O')
        cn = cn.replace('\u014d', 'O')
        cn = cn.replace('\u016b', 'U')
        cn = cn.replace('\u2018', '')
    return cn


# Get unemployment rates by County (one-year averages)
def load_bls():
    link = 'http://www.bls.gov/lau/laucntycur14.txt'
    cols = ['Code', 'StateFIPS', 'CountyFIPS', 'County', 
    	'Period', 'CLF', 'Employed', 'Unemployed', 'Rate']
    file = requests.get(link)
    rows = [l.split('|') for l in file.text.split('\r\n') if l.startswith(' CN')]
    data =pd.DataFrame(rows, columns=cols)
    data['Period'] = data['Period'].apply(lambda x:dt.strptime(x.strip()[:6],'%b-%y'))

    # keep only most recent 12 months; note np.unique also sorts
    min_date = np.unique(data['Period'])[2]
    data = data[data['Period']>=min_date]

    # reduce Code to just state/county number
    to_float = lambda x: float(str(x).replace(',',''))
    data['Code'] = data['Code'].apply(lambda x: x.strip()[2:7])

    # convert numerical data to floats
    for col in ['CLF', 'Employed', 'Unemployed']:
        data[col] = data[col].apply(to_float)
    data['Rate'] = data['Rate'].astype(float)
 
    grp = data.groupby('Code')
    summary = grp.mean()
    summary['County'] = grp.first()['County'].apply(lambda x:str(x).strip())
    
    return summary
    
####

z2f = json.load(open(os.path.join(parent_dir, 'z2f.json'), 'r'))
data = load_bls()
        
