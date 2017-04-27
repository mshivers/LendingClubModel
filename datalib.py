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


class LocationDataManager(object):
    ''' Manages the current BLS, FHFA, and other reference data'''
    def __init__(self):
        self.bls_data = CurrentBLSData() 
        self.reference_data = ReferenceData()
        self.fhfa_data = FHFAData() 

    def get(self, zip3, state):
        ''' takes the first three digits of the zip code and returns
        a dictionary of features for that location'''
        info = dict()
        ur, avg_ur, ur_chg = self.bls_data.get_urate(zip3)
        info['urate'] = ur
        info['avg_urate'] = avg_ur
        info['urate_chg'] = ur_chg
        info['census_median_income'] = self.reference_data.get_median_income(zip3)
        info['primaryCity'] = self.reference_data.get_primary_city(zip3)
        cbsa_list = self.reference_data.get_cbsa_list(zip3)
        metro_hpa = self.fhfa_data.get_metro_data().ix[cbsa_list].dropna()
        if len(metro_hpa)>0:
            info['hpa4'] = metro_hpa['1yr'].mean()
            info['hpa20'] = metro_hpa['5yr'].mean()
        else:
            print 'No FHFA data found for zip code {}xx, using data for {} instead'.format(zip3, state)
            nonmetro_hpa = self.fhfa_data.get_nonmetro_data().ix[state]
            info['hpa4'] = nonmetro_hpa['1yr']
            info['hpa20'] = nonmetro_hpa['5yr']
        return info


class FHFAData(object):
    _data_dir = PathManager.get_dir('fhfa')
    _metro_data = None
    _nonmetro_data = None

    def __init__(self):
        self._load_metro_data()
        self._load_nonmetro_data()

    @classmethod
    def get_metro_data(cls):
        return cls._metro_data

    @classmethod
    def get_nonmetro_data(cls):
        return cls._nonmetro_data

    @classmethod
    def _load_nonmetro_data(cls):
        ''' Loads the state-wide nonmetro area house price index data from the FHFA'''
        if cls._nonmetro_data is None:
            link='http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_nonmetro.xls'
            try:
                data = pd.read_excel(link, skiprows=2)
                data.to_csv(os.path.join(cls._data_dir, 'HPI_AT_nonmetro.csv'))
            except:
                data = pd.read_csv(os.path.join(cls._data_dir,'HPI_AT_nonmetro.csv'))
                print '{}: Failed to load FHFA nonmetro data; using cache\n'.format(dt.now())

            grp = data.groupby('State')
            tail5 = grp.tail(21).groupby('State')['Index']
            chg5 = np.log(tail5.last()) - np.log(tail5.first())
            tail1 = grp.tail(5).groupby('State')['Index']
            chg1 = np.log(tail1.last()) - np.log(tail1.first())
            chg = 100.0 * pd.DataFrame({'1yr':chg1, '5yr':chg5})
            cls._nonmetro_data = chg 

    @classmethod
    def _load_metro_data(cls):
        ''' Loads the FHFA metro area home price quarterly index data for Census Bureau
        Statistical Areas (CBSA) '''
        if cls._metro_data is None:
            link = "http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_metro.csv"
            cols = ['Location','CBSA', 'yr','qtr','index', 'stdev']
            try:
                data = pd.read_csv(link, header=None, names=cols)
                data.to_csv(os.path.join(cls._data_dir, 'HPI_AT_metro.csv'))
            except:
                data = pd.read_csv(os.path.join(cls._data_dir,'HPI_AT_metro.csv'), skiprows=1, header=None, names=cols)
                print '{}: Failed to load FHFA metro data; using cache\n'.format(dt.now())
            data = data[data['index']!='-']
            data['index'] = data['index'].astype(float)
            grp = data.groupby('CBSA')[['Location','CBSA', 'yr','qtr','index', 'stdev']]
            tail5 = grp.tail(21).groupby(['CBSA','Location'])['index']
            chg5 = np.log(tail5.last()) - np.log(tail5.first())
            tail1 = grp.tail(5).groupby(['CBSA','Location'])['index']
            chg1 = np.log(tail1.last()) - np.log(tail1.first())
            chg = 100.0 * pd.DataFrame({'1yr':chg1, '5yr':chg5})
            chg = chg.reset_index(1)
            cls._metro_data = chg

    
# Get unemployment rates by County (one-year averages)
class CurrentBLSData(object):
    def __init__(self):
        data_dir = PathManager.get_dir('bls')
        self.bls_fname = os.path.join(data_dir, 'bls_summary.csv')
        self.bls_data = None
        if self._cache_age() > 15:
            self._load_from_bls_website()
        if self.bls_data is None:
            self._load_from_cache()

    def get_urate(self, zip3):
        if zip3 in self.bls_data.index:
            urate = self.bls_data.ix[zip3,'current']
            ur_chg = self.bls_data.ix[zip3,'chg12m']
            avg_ur = self.bls_data.ix[zip3,'avg']
        else:
            nearby_zip3_idx = np.argmin(np.abs(np.array(bls.index)-zip3))
            near_zip3 = self.bls_data.index[nearby_zip3_idx]
            urate = self.bls_data.ix[near_zip3,'current'].mean()
            ur_chg = self.bls_data.ix[near_zip3,'chg12m'].mean()
            avg_ur = self.bls_data.ix[near_zip3,'avg'].mean()
        return urate, avg_ur, ur_chg

    def _cache_age(self):
        if os.path.exists(self.bls_fname):
            update_dt = dt.fromtimestamp(os.path.getmtime(self.bls_fname))
            days_old = (dt.now() - update_dt).days 
        else:
            days_old = np.inf
        return days_old

    def _load_from_cache(self):
        if os.path.exists(self.bls_fname):
            summary = pd.read_csv(self.bls_fname, index_col=0)
            self.bls_data = summary

    def _load_from_bls_website(self):
        try:
            ''' This loads the monthly employment data for the trailing 14 months '''
            print 'Downloading BLS data from bls.gov'
            link = 'http://www.bls.gov/lau/laucntycur14.txt'
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
            z2f = load_z2f()
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
            summary.to_csv(self.bls_fname)
            self.bls_data = summary 

        except:
            print '{}: Failed to load BLS laucntycur14 data; using summary cache\n'.format(dt.now())
    

class ReferenceData(object):
    ''' This class manages access to the data in the reference data directory,
    which contains fixed data (not regularly updated) by zip code regions '''
    _cache_dir = PathManager.get_dir('reference')
    _loanstats_to_api = None 
    _zip3_to_cbsa = None 
    _zip3_to_location_name = None 
    _zip3_to_primary_city = None 
    _zip3_to_fips = None
    _zip3_to_median_income = None

    def __init__(self):
        self.load_loanstats_to_api()
        self.load_zip3_to_cbsa()
        self.load_zip3_to_location_name()
        self.load_zip3_to_primary_city()
        self.load_zip3_to_fips()
        self.load_census_median_income()

    @classmethod
    def load_loanstats_to_api(cls):
        '''This maps the field names in the LendingClub historical data to the analogous name
        in the API data, if the analogous data exists'''
        if cls._loanstats_to_api is None:
            col_name_file = open(os.path.join(cls._cache_dir, 'loanstats2api.txt'), 'r')
            col_name_map = dict([line.strip().split(',') for line in col_name_file.readlines()])
            cls._loanstats_to_api = col_name_map
        return 
    
    @classmethod
    def get_loanstats2api_map(cls):
        return cls._loanstats_to_api

    @classmethod
    def load_zip3_to_cbsa(cls):
        if cls._zip3_to_cbsa is None:
            ''' Core crosswalk file from www.huduser.org/portal/datasets/usps_crosswalk.htlm'''
            data = pd.read_csv(os.path.join(cls._cache_dir, 'z2c.csv'))
            data['3zip'] = (data['ZIP']/100).astype(int)
            data['CBSA'] = data['CBSA'].astype(int)
            data = data[data['CBSA']!=99999]
            grp = data.groupby('3zip')
            z2c = defaultdict(lambda :list())
            for z in grp.groups.keys():
                z2c[z] = list(set(grp.get_group(z)['CBSA'].values))
            cls._zip3_to_cbsa = z2c
        return  

    @classmethod
    def get_cbsa_list(cls, zip3):
        return cls._zip3_to_cbsa[zip3]

    @classmethod
    def load_zip3_to_location_name(cls):
        ''' loads a dictionary mapping the first 3 digits of the zip code to a 
        list of location names''' 
        if cls._zip3_to_location_name is None:
            data = json.load(open(os.path.join(cls._cache_dir, 'zip2location.json'), 'r'))
            cls._zip3_to_location_name = dict([int(k), locs] for k, locs in data.items())
        return

    @classmethod
    def get_primary_city(cls, zip3):
        return cls._zip3_to_primary_city[zip3]

    @classmethod
    def load_zip3_to_primary_city(cls):
        ''' loads a dictionary mapping the first 3 digits of the zip code to the
        most common city with that zip code prefix'''
        if cls._zip3_to_primary_city is None:
            data = json.load(open(os.path.join(cls._cache_dir,'zip2primarycity.json')))
            cls._zip3_to_primary_city = dict([int(k), locs] for k, locs in data.items())
        return

    @classmethod
    def load_zip3_to_fips(cls):
        ''' loads a dictionary mapping the first 3 digits of the zip code to a 
        list of FIPS codes'''
        if cls._zip3_to_fips is None:
            d = json.load(open(os.path.join(cls._cache_dir, 'z2f.json'), 'r'))
            cls._zip3_to_fips = dict([int(k), [int(v) for v in vals]] for k, vals in d.items())
        return

    def _get_income_data(self, census, fips_list):
        # Returns the percentage of census tracts with incomes below $5000/mth, and
        # the median census tract income for the input FIPS codes
        # if the census tract doesn't exist (for overseas military, for example), 
        # return the average for the country.
        df = census[census.FIPS.isin(fips_list)]
        if len(df) > 0:
            result = df.CENINC.median()
        else:
            result = census.CENINC.median()
        
        return result

    @classmethod
    def load_census_median_income(cls):
        if cls._zip3_to_median_income is None:
            z2mi = json.load(open(os.path.join(cls._cache_dir, 'zip2median_inc.json'),'r'))
            z2mi = dict([(int(z), float(v)) for z,v in zip(z2mi.keys(), z2mi.values())])
            z2mi = defaultdict(lambda :np.mean(z2mi.values()), z2mi)
            cls._zip3_to_median_income = z2mi

    @classmethod
    def get_median_income(cls, zip3):
        return cls._zip3_to_median_income[zip3]

    @classmethod
    def load_census_data(cls):
        ''' this dataset contains the census bureaus median income data and 
        percentage poverty by FIPS code '''
        if cls._census_data is None:
            fname = os.path.join(cls._cache_dir, 'lya2014.txt')
            rows = open(fname,'r').read().split('\r\n')
            data = [r.split() for r in rows]
            df = pd.DataFrame(data[1:], columns=data[0])
            df = df.dropna()
            df['STATE'] = df['STATE'].astype(int)
            df['CNTY'] = df['CNTY'].astype(int)
            df['FIPS'] = 1000 * df['STATE'] + df['CNTY']
            df['LYA'] = df['LYA'].astype(float)
            df['CENINC'] = df['CENINC'].astype(float)
            df = df[df['LYA'] <= 1]  # remove missing data (LYA==9)
            cls._census_data = df

    @staticmethod
    def load_f2c(cache_dir):
        ''' loads a dictionary mapping FIPS codes to CBSA codes and names '''
        #need a mapping that lists NY-NJ to 35614, rather than 36544
        d = json.load(open(os.path.join(cache_dir, 'fips2cbsa.json'), 'r'))
        return dict([int(k), [int(v[0]), v[1]]] for k, v in d.items())

    @staticmethod
    def construct_z2c(z2f, f2c):
        z2c = dict()
        for z, fips in z2f.iteritems():
            clist = set()
            for f in fips:
                if f in f2c.keys():
                   clist.add(f2c[f][0])
            z2c[z] = clist 
        return z2c 

    @staticmethod
    def build_zip3_to_location_names(cache_dir):
        import bls 
        cw = pd.read_csv(os.path.join(cache_dir, 'CBSA_FIPS_MSA_crosswalk.csv'))
        grp = cw.groupby('FIPS')
        f2loc = dict([(k,list(df['CBSA Name'].values)) 
                      for k in grp.groups.keys()
                      for df in [grp.get_group(k)]])

        z3f = json.load(open(os.path.join(cache_dir, 'zip3_fips.json'),'r'))
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
        json.dump(z2loc, open(os.path.join(cache_dir, 'zip2location.json'),'w'))

    @staticmethod
    def _build_zip3_to_primary_city(cache_dir):
        data= pd.read_csv(os.path.join(cache_dir, 'zip_code_database.csv'))
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

        json.dump(z2primarycity, open(os.path.join(cache_dir, 'zip2primarycity.json'),'w'))


def get_external_data():
    #def build_zip3_to_hpi():
    z2c = pd.read_csv(os.path.join(reference_data_dir, 'zip2cbsa.csv'))
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
        data.to_csv(os.path.join(fhfa_data_dir, 'HPI_AT_metro.csv'))
    except:
        print 'Failed to read FHFA website HPI data; using cached data'
        data = pd.read_csv(os.path.join(fhfa_data_dir,'HPI_AT_metro.csv'), header=None, names=cols)

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

    hpa1.to_csv(os.path.join(fhfa_data_dir,'hpa1.csv'))
    hpa4.to_csv(os.path.join(fhfa_data_dir,'hpa4.csv'))
    hpa20.to_csv(os.path.join(fhfa_data_dir,'hpa20.csv'))
    hpa40.to_csv(os.path.join(fhfa_data_dir,'hpa40.csv'))

    '''
    # get non-metro hpi, for other zip codes
    link='http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_nonmetro.xls'
    try:
        data = pd.read_excel(link, skiprows=2)
        data.to_csv(os.path.join(p.parent_dir, 'HPI_AT_nonmetro.csv'))
    except:
        data = pd.read_csv(os.path.join(p.parent_dir,'HPI_AT_nonmetro.csv'))

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
    z2f = json.load(file(os.path.join(reference_data_dir, 'zip3_fips.json'),'r'))

    #z2f values are lists themselves; this flattens it
    all_fips = []
    for f in z2f.values():
        all_fips.extend(f) 
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
    labor_force.to_csv(os.path.join(bls_data_dir, 'labor_force.csv'))
    unemployed.to_csv(os.path.join(bls_data_dir, 'unemployed.csv'))

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
    urate.to_csv(os.path.join(bls_data_dir, 'urate_by_3zip.csv'))



def build_zip3_to_location_names():
    import bls 
    reference_data_dir = paths.get_dir('reference')
    cw = pd.read_csv(os.path.join(reference_data_dir, 'CBSA_FIPS_MSA_crosswalk.csv'))
    grp = cw.groupby('FIPS')
    f2loc = dict([(k,list(df['CBSA Name'].values)) 
                  for k in grp.groups.keys()
                  for df in [grp.get_group(k)]])

    z3f = json.load(open(os.path.join(reference_data_dir, 'zip3_fips.json'),'r'))
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

    json.dump(z2loc, open(os.path.join(reference_data_dir, 'zip2location.json'),'w'))

def build_zip3_to_primary_city():
    reference_data_dir = paths.get_dir('reference')
    data= pd.read_csv(os.path.join(reference_data_dir, 'zip_code_database.csv'))
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

    json.dump(z2primarycity, open(os.path.join(reference_data_dir, 'zip2primarycity.json'),'w'))


def get_external_data():
    #def build_zip3_to_hpi():
    reference_data_dir = paths.get_dir('reference')
    z2c = pd.read_csv(os.path.join(reference_data_dir, 'zip2cbsa.csv'))
    z2c['zip3'] = z2c['ZIP'].apply(lambda x: int(x/100))
    z2c = z2c[z2c['CBSA']<99999]
    grp = z2c.groupby('zip3')

    z2clist = dict()
    for z3 in grp.groups.keys():
        g = grp.get_group(z3)
        z2clist[z3] = sorted(list(set(g['CBSA'].values)))


    # get metro hpi for main areas
    fhfa_data_dir = paths.get_dir('fhfa')
    link = "http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_metro.csv"
    cols = ['Location','CBSA', 'yr','qtr','index', 'stdev']
    try:
        data = pd.read_csv(link, header=None, names=cols)
        data.to_csv(os.path.join(fhfa_data_dir, 'HPI_AT_metro.csv'))
    except:
        print 'Failed to read FHFA website HPI data; using cached data'
        data = pd.read_csv(os.path.join(fhfa_data_dir,'HPI_AT_metro.csv'), header=None, names=cols)

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

    hpa1.to_csv(os.path.join(fhfa_data_dir,'hpa1.csv'))
    hpa4.to_csv(os.path.join(fhfa_data_dir,'hpa4.csv'))
    hpa20.to_csv(os.path.join(fhfa_data_dir,'hpa20.csv'))
    hpa40.to_csv(os.path.join(fhfa_data_dir,'hpa40.csv'))

    '''
    # get non-metro hpi, for other zip codes
    link='http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_nonmetro.xls'
    try:
        data = pd.read_excel(link, skiprows=2)
        data.to_csv(os.path.join(p.parent_dir, 'HPI_AT_nonmetro.csv'))
    except:
        data = pd.read_csv(os.path.join(p.parent_dir,'HPI_AT_nonmetro.csv'))

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
    z2f = json.load(file(os.path.join(reference_data_dir, 'zip3_fips.json'),'r'))

    #z2f values are lists themselves; this flattens it
    all_fips = []
    for f in z2f.values():
        all_fips.extend(f) 
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

    bls_data_dir = paths.get_dir('bls')
    labor_force.to_csv(os.path.join(bls_data_dir, 'labor_force.csv'))
    unemployed.to_csv(os.path.join(bls_data_dir, 'unemployed.csv'))

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
    urate.to_csv(os.path.join(bls_data_dir, 'urate_by_3zip.csv'))
        


