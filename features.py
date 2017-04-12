import datalib
from sklearn.externals import joblib
from collections import defaultdict
from collections import Counter
from datetime import datetime as dt
import numbers
import pandas as pd
import numpy as np
import json
import os

class FeatureManager(object):
    '''
    FeatureManager is responsible for processing the validated API loan data and fully
    populating the loan dictionary with all fields required by the prepayment and 
    default models.
    '''

    def __init__(self, model_dir=None):
        self.model_dir = model_dir
        self.location_features = datalib.LocationDataManager()
        self.simple_features = SimpleFeatures()
        self.feature_models = list()
        self.randomforests= list()
        self.load_models()

    def load_models(self):
        if self.model_dir is not None:
            model_files = os.listdir(self.model_dir)
            for fname in model_files:
                if fname.endswith('_odds.json'):
                    config_path = os.path.join(self.model_dir, fname)
                    config = json.load(open(config_path, 'r'))
                    feature = OddsFeature(**config)
                    self.feature_models.append(feature)
                    print 'Loaded {} model'.format(feature.feature_name)
                if fname.endswith('_frequency.json'):
                    config_path = os.path.join(self.model_dir, fname)
                    config = json.load(open(config_path, 'r'))
                    feature = FrequencyFeature(**config) 
                    self.feature_models.append(feature)
                    print 'Loaded {} model'.format(feature.feature_name)
                if fname.endswith('_randomforest.json'):
                    config_path = os.path.join(self.model_dir, fname)
                    config = json.load(open(config_path, 'r'))
                    pkl_file = config.pop('pkl_filename')
                    forest_path = os.path.join(self.model_dir, pkl_file)
                    feature = RandomForestFeature(pkl_file=forest_path, **config) 
                    self.randomforests.append(feature)
                    print 'Loaded {} model'.format(feature.feature_name)
             
    def process_lookup_features(self, loan):
        ''' these features are simple dictionary lookup features, and quick'''
        features = self.location_features.get(loan['addrZip'], loan['addrState'])
        loan.update(features)
        self.simple_features.calc(loan)
        for model in self.feature_models:
            loan[model.feature_name] = model.calc(loan[model.string_name])

    def process_forest_features(self, loan): 
        ''' these require running a random forest, so they're slow '''
        for model in self.randomforests:
            loan[model.feature_name] = model.calc(loan)


class FrequencyFeature(object):
    ''' Model that converts a string feature into the number of 
    instances of that string that exists in the historical data'''

    def __init__(self, freq_dict=None, string_name=None):
        if isinstance(freq_dict, dict):
            freq_dict = defaultdict(lambda :0, freq_dict)
        self.freq_dict = freq_dict
        self.string_name = string_name
        self.feature_name = '{}_frequency'.format(self.string_name)

    def fit(self, strings):
        '''Creates the dictionary with the frequency mapping'''
        counts = Counter(strings)
        self.freq_dict = defaultdict(lambda :0, counts.items())

    def calc(self, input_string):
        return self.freq_dict[input_string]

    def is_fit(self):
        return isinstance(self.freq_dict, dict)

    def save(self, fname):
        config = {'freq_dict': dict(self.freq_dict), 
                  'string_name':self.string_name}
        if os.path.isdir(fname):
            file_name = '{}.json'.format(self.feature_name)
            fname = os.path.join(fname, file_name) 
        json.dump(config, open(fname, 'w'), indent=4)


class OddsFeature(object):
    def __init__(self, tok_type, odds_dict=None, string_name=None, value_name=None):
        self.tok_type = tok_type
        if isinstance(odds_dict, dict):
            odds_dict = defaultdict(lambda :0, odds_dict)
        self.odds_dict = odds_dict 
        self.string_name = string_name #name of the string field to apply the model to.
        self.value_name = value_name #name of the value field the model predicts 
        self.feature_name = '{}_{}_{}_odds'.format(self.value_name, self.string_name, self.tok_type)

    @staticmethod
    def _get_all_substrings(input_string):
        length = len(input_string)
        return [input_string[i:j+1] for i in xrange(length) for j in xrange(i,length)]
       
    @staticmethod
    def _get_short_substrings(input_string, max_len):
        length = len(input_string)
        return [input_string[i:j+1] for i in xrange(length) 
                for j in xrange(i+1,length) if j-i<max_len]
       
    def _get_exptok_substrings(self, input_string):
        toks = list()
        for i in range(int(np.log2(len(input_string)))+1):
            toks.extend(self._get_substrings_of_length(input_string, 2**i)) 
            if input_string not in toks:
                toks.append(input_string)
        return toks

    @staticmethod
    def _get_substrings_of_length(input_string, length):
        return [input_string[i:i+length] for i in range(max(1,len(input_string)-length))]

    @staticmethod
    def _get_words(self, input_string):
        return input_string.strip().split()

    def get_tokens(self, input_string):
        if self.tok_type=='word':
            toks = self._get_words(input_string) 
        elif self.tok_type=='phrase':
            toks = [input_string]
        elif self.tok_type=='alltoks':
            toks = self._get_all_substrings(input_string) 
        elif self.tok_type=='shorttoks':
            toks = self._get_short_substrings(input_string, 10) 
        elif self.tok_type=='exptoks':
            toks = self._get_exptok_substrings(input_string)
        elif isinstance(self.tok_type, int):
            toks = self._get_substrings_of_length(input_string, self.tok_type) 
        else:
            raise Exception('unknown tok_type')
        return set(toks)

    def fit(self, strings, values):
        ''' creates odds dictionary.  The tok_type field is either the length of the string token
        to use, or if tok_type='word', the tokens are entire words'''
        value_sum = defaultdict(lambda :0)
        tok_count = defaultdict(lambda :0)
        data = pd.DataFrame({'strings':strings, 'values':values})
        grp = data.groupby('strings')
        summary = grp.agg(['sum', 'count'])['values']
        count = Counter()
        for string, row in summary.iterrows():
            tokens = self.get_tokens(string)
            val_incr = row['sum']
            count_incr = row['count']
            for tok in tokens:
                value_sum[tok] += val_incr 
                tok_count[tok] += count_incr 

        C = 500.0
        tok_count = pd.Series(tok_count)
        value_sum = pd.Series(value_sum)
        global_mean = data['values'].mean()
        tok_mean = (value_sum + C * global_mean) / (tok_count + C)
        odds = tok_mean - global_mean

        # the random forest to overfit to those.
        odds = odds[tok_count>100]
        self.odds_dict = defaultdict(lambda :0, odds.to_dict())

    def calc(self, input_string):
        toks = self.get_tokens(input_string)
        odds = np.sum(map(lambda x:self.odds_dict[x], toks))
        return odds

    def tok_odds_array(self, input_string):
        toks = self.get_tokens(input_string)
        odds = np.array(map(lambda x:self.odds_dict[x], toks))
        return odds

    def is_fit(self):
        return isinstance(self.odds_dict, dict)

    def save(self, fname):
        config = {'odds_dict': dict(self.odds_dict), 
                  'string_name':self.string_name, 
                  'value_name':self.value_name, 
                  'tok_type': self.tok_type}
        if os.path.isdir(fname):
            file_name = '{}.json'.format(self.feature_name)
            fname = os.path.join(fname, file_name) 
        json.dump(config, open(fname, 'w'), indent=4)

  
class SimpleFeatures(object):
    feature_names = ['credit_length', 'int_pymt', 'revol_bal-loan']
    required_features = ['earliestCrLine', 'loanAmount']

    def __init__(self):
        pass

    def _required_fields_exist(self, data):
        if isinstance(data, dict):
            return all([fld in data.keys() for fld in self.required_features]) 

    def calc(self, data):
        if isinstance(data, pd.DataFrame):   #historical data
            one_year = 365*24*60*60*1e9
            data['credit_length'] = ((data['issue_d'] - data['earliestCrLine']).astype(int)/one_year)
            data['credit_length'] = data['credit_length'].apply(lambda x: max(-1,round(x,0)))
            data['even_loan_amnt'] = data['loanAmount'].apply(lambda x: float(x==round(x,-3)))
        else:   #API data
            earliest_credit = dt.strptime(data['earliestCrLine'].split('T')[0],'%Y-%m-%d')
            seconds_per_year = 365*24*60*60.0
            data['credit_length'] = (dt.now() - earliest_credit).total_seconds() / seconds_per_year
            data['even_loan_amnt'] = float(data['loanAmount'] == np.round(data['loanAmount'],-3))

        data['int_pymt'] = data['loanAmount'] * data['intRate'] / 1200.0
        data['revol_bal-loan'] = data['revolBal'] - data['loanAmount']
        data['inc_pct_med_inc'] = data['annualInc'] / data['census_median_income']
        data['pymt_pct_inc'] = data['installment'] / data['annualInc'] 
        data['revol_bal_pct_inc'] = data['revolBal'] / data['annualInc']
        data['int_pct_inc'] = data['int_pymt'] / data['annualInc'] 
        data['loan_pct_income'] = data['loanAmount'] / data['annualInc']
        data['cur_bal-loan_amnt'] = data['totCurBal'] - data['loanAmount'] 
        data['cur_bal_pct_loan_amnt'] = data['totCurBal'] / data['loanAmount'] 
        data['mort_bal'] = data['totCurBal'] - data['totalBalExMort']
        data['mort_pct_credit_limit'] = data['mort_bal'] * 1.0 / data['totHiCredLim']
        data['mort_pct_cur_bal'] = data['mort_bal'] * 1.0 / data['totCurBal']
        data['revol_bal_pct_cur_bal'] = data['revolBal'] * 1.0 / data['totCurBal']
        data['empTitle_length'] = len(data['empTitle']) 


class RandomForestFeature(object):
    def __init__(self, pkl_file, inputs, feature_name, pctl):
        self.pkl_file = pkl_file
        self.random_forest = joblib.load(pkl_file) 
        self.random_forest.verbose=0
        self.input_fields = inputs
        self.feature_name = feature_name
        self.pctl = pctl
 
    def required_inputs(self):
        return self.input_fields

    def validate_input_data(self, input_data):
        valid = True
        if not isinstance(input_data, dict):
            print 'RandomForest Input Error: input must be a dict, not a {}'.format(type(input_data))
            valid = False

        for field in self.required_inputs():
            if field in input_data.keys():
                value = input_data[field]
                if not (isinstance(value, numbers.Number) and np.isfinite(value)):
                    print 'RandomForest Invalid Input Error: {}:{} not valid'.format(field, value)
                    valid = False
            else:
                print 'RandomForest Missing Input Error: {} not found'.format(field)
                valid = False

        return valid 

    def calc(self, input_data):
        valid = self.validate_input_data(input_data)
        if valid: 
            x = np.zeros(len(self.input_fields)) * np.nan
            for i, fld in enumerate(self.input_fields):
                x[i] = input_data[fld]

            predictions = [tree.predict(x)[0] for tree in self.random_forest.estimators_]
            prediction =  np.percentile(predictions, self.pctl)
        
        else: 
            prediction = np.nan

        return prediction 


