import os
import json
import requests
import numbers
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from collections import defaultdict, Counter
from sklearn.externals import joblib
import scipy.signal
from matplotlib import pyplot as plt
import utils
from personalized import p
import datalib
from constants import PathManager as paths

def load_training_data(regen=False):
    fname = paths.get_file('training')
    if os.path.exists(fname) and not regen:
        update_dt = dt.fromtimestamp(os.path.getmtime(fname))
        days_old = (dt.now() - update_dt).days 
        print 'Using cached LC data created on {}; cache is {} days old'.format(update_dt, days_old)
        df = pd.read_csv(fname, header=0, index_col=0)
    else:
        print 'Cache not found. Generating cache from source data'
        cache_base_data()
        update_training_data()
        df = load_training_data()
    return df

def cache_base_data():
    '''This assembles the LC data and adds standard fields that don't change'''
    #rename columns to match API fields
    ref_data = datalib.ReferenceData() 
    col_name_map = ref_data.get_loanstats2api_map()

    def clean_raw_data(df):
        df = df.rename(columns=col_name_map)
        idx1 = ~(df[['last_pymnt_d', 'issue_d', 'annualInc']].isnull()).any(1)
        df = df.ix[idx1].copy()
        df['issue_d'] = df['issue_d'].apply(lambda x: dt.strptime(x, '%b-%Y'))
        idx2 = df['issue_d']>=np.datetime64('2013-10-01')
        idx3 = df['issue_d'] <= dt.now() - td(days=366)
        df = df.ix[idx2&idx3].copy()
        df['id'] = df['id'].astype(int)
        return df

    loanstats_dir = paths.get_dir('loanstats')
    fname = os.path.join(loanstats_dir, 'LoanStats3{}_securev1.csv')
    fname2 = os.path.join(loanstats_dir, 'LoanStats_securev1_{}.csv')
    dataframes = list()
    print 'Importing Raw Data Files'
    dataframes.append(clean_raw_data(pd.read_csv(fname.format('a'), header=1)))
    dataframes.append(clean_raw_data(pd.read_csv(fname.format('b'), header=1)))
    dataframes.append(clean_raw_data(pd.read_csv(fname.format('c'), header=1)))
    dataframes.append(clean_raw_data(pd.read_csv(fname.format('d'), header=1)))
    dataframes.append(clean_raw_data(pd.read_csv(fname2.format('2016Q1'), header=1)))
    #dataframes.append(clean_raw_data(pd.read_csv(fname2.format('2016Q2'), header=1)))
    #dataframes.append(clean_raw_data(pd.read_csv(fname2.format('2016Q3'), header=1)))
    print 'Concatenating dataframes'
    df = pd.concat(dataframes, ignore_index=True)
    print 'Dataframes imported'
 
    # clean dataframe
    cvt = dict()
    # intRate and revolUtil are the only fields where the format is "xx.x%" (with a % sign in the string)
    cvt['intRate'] = lambda x: float(x[:-1])
    cvt['revolUtil'] = lambda x: np.nan if '%' not in str(x) else round(float(x[:-1]),0)
    cvt['desc'] = lambda x: float(len(str(x)) > 3)
    cvt['last_pymnt_d'] = lambda x: dt.strptime(x, '%b-%Y')
    cvt['earliestCrLine'] = lambda x: dt.strptime(x, '%b-%Y')
    cvt['term'] = lambda x: int(x.strip().split(' ')[0])

    for col in cvt.keys():
        print 'Parsing {}'.format(col)
        df[col] = df[col].apply(cvt[col])

    api_parser = datalib.APIDataParser()
    for field in api_parser.null_fill_fields():
        if field in df.columns:
            fill_value = api_parser.null_fill_value(field)
            print 'Filling {} nulls with {}'.format(field, fill_value)
            df[field] = df[field].fillna(fill_value)

    string_converter = datalib.StringToConst()
    for col in string_converter.accepted_fields:
        if col in df.columns:
            print 'Converting {} string to numeric'.format(col)
            func = np.vectorize(string_converter.convert_func(col))
            df[col] = df[col].apply(func)
   
    df['empTitle'] = df['empTitle'].apply(utils.format_title)
    df['empTitle_length'] = df['empTitle'].apply(lambda x: len(x))

    # Calculate target values for various prediction models
    # add default info
    print 'Calculating Target model target values'
    df['wgt_default'] = df['loan_status'].apply(DefaultProb.by_status) 

    # we want to find payments strictly less than 1 year, so we use 360 days here.
    just_under_one_year = 360*24*60*60*1e9  
    time_to_last_pymnt = df['last_pymnt_d']-df['issue_d']
    df['12m_late'] = (df['wgt_default']>0) & (time_to_last_pymnt<just_under_one_year)
    df['12m_wgt_default'] = df['12m_late'] * df['wgt_default']

    # add prepayment info
    df['12m_prepay'] = 0.0
    # for Fully Paid, assume all prepayment happened in last month
    just_over_12months = 12.5*30*24*60*60*1e9  
    prepay_12m_idx = ((df['loan_status']=='Fully Paid') & (time_to_last_pymnt<=just_over_12months))
    df.ix[prepay_12m_idx, '12m_prepay'] = 1.0

    # partial prepays
    df['mob'] = np.ceil(time_to_last_pymnt.astype(int) / (just_over_12months / 12.0))
    prepayments = np.maximum(0, df['total_pymnt'] - df['mob'] * df['installment'])
    partial_12m_prepay_idx = (df['loan_status']=='Current') & (prepayments > 0)
    prepay_12m_pct = prepayments / df['loanAmount'] * (12. / np.maximum(12., df.mob))
    df.ix[partial_12m_prepay_idx, '12m_prepay'] = prepay_12m_pct[partial_12m_prepay_idx]

    ### Add non-LC features
    print 'Adding BLS data'
    bls_data_dir = paths.get_dir('bls')
    urate = pd.read_csv(os.path.join(bls_data_dir, 'urate_by_3zip.csv'), index_col=0)
    ur = pd.DataFrame(np.zeros((len(urate),999))*np.nan,index=urate.index, columns=[str(i) for i in range(1,1000)])
    ur.ix[:,:] = urate.median(1).values[:,None]
    ur.ix[:,urate.columns] = urate
    avg_ur = pd.rolling_mean(ur, 12)
    ur_chg = ur - ur.shift(12)

    df['urate_d'] = df['issue_d'].apply(lambda x: int(str((x-td(days=60)))[:7].replace('-','')))
    df['urate'] = [ur[a][b] for a,b in zip(df['addrZip'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['avg_urate'] = [avg_ur[a][b] for a,b in zip(df['addrZip'].apply(lambda x: str(int(x))), df['urate_d'])]
    df['urate_chg'] = [ur_chg[a][b] for a,b in zip(df['addrZip'].apply(lambda x: str(int(x))), df['urate_d'])]

    print 'Adding FHFA data'
    fhfa_data_dir = paths.get_dir('fhfa')
    hpa4 = pd.read_csv(os.path.join(fhfa_data_dir, 'hpa4.csv'), index_col = 0)
    mean_hpa4 = hpa4.mean(1)
    missing_cols = [str(col) for col in range(0,1000) if str(col) not in hpa4.columns]
    for c in missing_cols:
        hpa4[c] = mean_hpa4

    df['hpa_date'] = df['issue_d'].apply(lambda x:x-td(days=120))
    df['hpa_qtr'] = df['hpa_date'].apply(lambda x: 100*x.year + x.month/4 + 1)
    df['hpa4'] = [hpa4.ix[a,b] for a,b in zip(df['hpa_qtr'], df['addrZip'].apply(lambda x: str(int(x))))]
     
    print 'Adding Census data'
    df['census_median_income'] = df['addrZip'].apply(ref_data.get_median_income)

    # Add calculated features 
    print 'Adding Binary Features'
    binary_features = BinaryFeatures()
    binary_features.calc(df)

    # tag data for in-sample and oos (in sample issued at least 14 months ago. Issued 12-13 months ago is oos
    df['in_sample'] = df['issue_d'] < dt.now() - td(days=14*31)

    print 'Saving base file'
    df.to_csv(paths.get_file('base'))


def update_training_data(df=None):
    if df is None:
        df = pd.read_csv(paths.get_file('base'), header=0, index_col=0)
        print 'Dataframes imported'
  
    # process job title features
    print 'Adding empTitle prepay odds features (this takes a while)'
    training_data_dir = paths.get_dir('training')
    sample = (df.grade>=2) & (df.in_sample)
    odds = OddsFeature(tok_type='alltoks', string_name='empTitle', value_name='prepay')
    odds.fit(df.ix[sample, 'empTitle'].values, df.ix[sample, '12m_prepay'].values)
    odds.save(training_data_dir)
    feature_name = odds.feature_name() 
    df[feature_name] = df[odds.string_name].apply(odds.calc)

    # process job title features
    print 'Adding empTitle default odds features'
    sample = (df.grade>=2) & (df.in_sample)
    odds = OddsFeature(tok_type='alltoks', string_name='empTitle', value_name='default')
    odds.fit(df.ix[sample, 'empTitle'].values, df.ix[sample, '12m_wgt_default'].values)
    odds.save(training_data_dir)
    feature_name = odds.feature_name() 
    df[feature_name] = df[odds.string_name].apply(odds.calc)

    #process frequency features
    print 'Adding frequency features'
    freq = FrequencyFeature(string_name='empTitle')
    freq.fit(df.ix[df.in_sample, 'empTitle'].values)
    freq.save(training_data_dir)
    feature_name = freq.feature_name()
    df[feature_name] = df[freq.string_name].apply(freq.calc)
 
    print 'Saving cache file'
    df.to_csv(paths.get_file('training'))


