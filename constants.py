import os
from collections import defaultdict
from personalized import p
import numpy as np


LARGE_INT = 9999 
NEGATIVE_INT = -1 
UPDATE_HOURS = [9,13,17,21]  #Times new loans are posted in NY time

class DefaultProb(object):
    _probs = defaultdict(lambda :0, {'Charged Off': 1.0,
                                     'Default': 0.86,
                                     'Late (31-120 days)': 0.77,
                                     'Late (16-30 days)': 0.59,
                                     'In Grace Period': 0.28})

    def __init__(self):
        pass

    @classmethod
    def by_status(cls, status):
        return cls._probs[status] 


class PathManager(object):
    data_dir = os.path.join(p.parent_dir, 'data')
    loanstats_dir = os.path.join(p.parent_dir, 'data/loanstats')
    training_data_dir = os.path.join(p.parent_dir, 'data/training_data')
    prod_dir = os.path.join(p.parent_dir, 'data/prod')
    reference_data_dir = os.path.join(p.parent_dir, 'data/reference_data')
    bls_data_dir = os.path.join(p.parent_dir, 'data/bls_data')
    fhfa_data_dir = os.path.join(p.parent_dir, 'data/fhfa_data')
    saved_prod_data_dir = os.path.join(p.parent_dir, 'data/saved_prod_data')
    payments_file = os.path.join(loanstats_dir, 'PMTHIST_ALL_20171018.csv')
    cached_training_data_file = os.path.join(training_data_dir, 'cached_training_data.csv')
    base_data_file = os.path.join(training_data_dir, 'base_data.csv')
    loanstats_data_file = os.path.join(loanstats_dir, 'loanstats_data.csv')
    employer_data_file = os.path.join(loanstats_dir, 'scraped_data/backoffice_employers.csv')
    require_returns_json = os.path.join(saved_prod_data_dir, 'required_returns.json')

    def __init__(self):
        pass
   
    @classmethod
    def get_dir(cls, item):
        if item=='loanstats':
            return cls.loanstats_dir
        elif item=='training':
            return cls.training_data_dir
        elif item=='reference':
            return cls.reference_data_dir
        elif item=='bls':
            return cls.bls_data_dir
        elif item=='fhfa':
            return cls.fhfa_data_dir
        elif item=='training':
            return cls.training_data_dir
        elif item=='prod':
            return cls.prod_dir
        else:
            return -1

    @classmethod
    def get_file(cls, item):
        if item=='payments':
            return cls.payments_file
        elif item in ['training', 'training_cache', 'training_data']:
            return cls.cached_training_data_file
        elif item in ['base', 'base_data', 'base_cache']:
            return cls.base_data_file
        elif item in ['loanstats', 'loanstats_data', 'loanstats_cache']:
            return cls.loanstats_data_file
        elif item == 'employer_data': 
            return cls.employer_data_file
        elif item == 'required_returns': 
            return cls.require_returns_json
        else:
            return -1

paths = PathManager()


# clean dataframe
class StringToConst(object):
    def __init__(self):
        self.accepted_fields = ['homeOwnership',
                                'purpose',
                                'grade',
                                'subGrade',
                                'isIncV', 
                                'isIncVJoint',
                                'initialListStatus',
                                'empLength',
                                'addrZip',
                                'applicationType']
        self.home_map = dict([('ANY', 0), ('NONE',0), ('OTHER',0), 
                              ('RENT',1), ('MORTGAGE',2), ('OWN',3)])
        self.purpose_dict = defaultdict(lambda :np.nan)
        self.purpose_dict.update([('CREDIT_CARD', 0), ('CREDIT_CARD_REFINANCING', 0), 
                                  ('DEBT_CONSOLIDATION',1), 
                                  ('HOME_IMPROVEMENT',2), 
                                  ('CAR',3), ('CAR_FINANCING',3), 
                                  ('EDUCATIONAL',4), 
                                  ('HOUSE',5), ('HOME_BUYING',5),
                                  ('MAJOR_PURCHASE',6), 
                                  ('MEDICAL_EXPENSES',7), ('MEDICAL',7), 
                                  ('MOVING',8), ('MOVING_AND_RELOCATION',8), 
                                  ('OTHER',9),
                                  ('RENEWABLE_ENERGY',10), ('GREEN_LOAN',10),
                                  ('BUSINESS',11),('SMALL_BUSINESS',11),
                                  ('VACATION',12), 
                                  ('WEDDING',13)])
        grades = list('ABCDEFG')
        self.grade_map = defaultdict(lambda :np.nan, zip(grades, range(len(grades))))
        subgrades = ['{}{}'.format(l,n) for l in 'ABCDEFG' for n in range(1,6)]
        self.subgrade_map = defaultdict(lambda :np.nan, zip(subgrades, range(len(subgrades))))
        loanstats_verification_dict = dict([('Verified',2), ('Source Verified',1), ('Not Verified',0)]) 
        api_verification_dict = dict([('VERIFIED',2), ('SOURCE_VERIFIED',1), ('NOT_VERIFIED',0)])
        self.income_verification = loanstats_verification_dict
        self.income_verification.update(api_verification_dict)
        self.init_status_dict = dict([('f',0), ('F',0), ('w',1), ('W',1)])
        self.application_type_dict = dict([('DIRECT_PAY', -1), ('INDIVIDUAL', 0), ('JOINT', 1)])

    def _convert_empLength(self, value):
        value=value.replace('< 1 year', '0')
        value=value.replace('1 year','1')
        value=value.replace('10+ years', '11')
        value=value.replace('n/a', '-1')
        value=value.replace(' years', '')
        return int(value)
    
    def _convert_grade(self, value):
        return self.grade_map[value]

    def _convert_application_type(self, value):
        return self.application_type_dict[value.upper()]

    def _convert_homeOwnership(self, value):
        return self.home_map[value.upper()]

    def _convert_purpose(self, value):
        value = value.lower().replace(' ', '_')
        return self.purpose_dict[value.upper()]

    def _convert_subGrade(self, value):
        return self.subgrade_map[value]

    def _convert_inc_verification(self, value):
        return self.income_verification[value.upper()]

    def _convert_initialListStatus(self, value):
        return self.init_status_dict[value.upper()]

    def _convert_addrZip(self, value):
        return int(value[:3])

    def convert(self, field, value):
        if field == 'homeOwnership':
            if value.upper() in self.home_map.keys():
                return self.home_map[value.upper()]
        elif field == 'purpose':
            value = value.replace(' ', '_').upper()
            if value in self.purpose_dict.keys():
                return self.purpose_dict[value]
        elif field == 'applicationType':
            value = value.replace(' ', '_').upper()
            if value in self.application_type_dict.keys():
                return self.application_type_dict[value]
            else:
                import pdb
                pdb.set_trace()
        elif field == 'grade':
            if value in self.grade_map.keys():
                return self.grade_map[value]
        elif field == 'subGrade':
            if value in self.subgrade_map.keys():
                return self.subgrade_map[value]
        elif field in ['isIncV', 'isIncVJoint']:
            if value in self.income_verification.keys():
                return self.income_verification[value]
        elif field == 'initialListStatus':
            if value in self.init_status_dict.keys():
                return self.init_status_dict[value]
        elif field == 'empLength':
            return self._convert_empLength(value)
        elif field == 'addrZip':
            return int(value[:3])
        else:
            return value
    
    def convert_func(self, field):
        if field == 'homeOwnership':
            return lambda x: self.home_map[x]
        elif field == 'purpose':
            return lambda x: self.purpose_dict[x.upper().replace(' ', '_')]
        elif field == 'applicationType':
            return lambda x: self.application_type_dict[x.upper()]
        elif field == 'grade':
            return lambda x: self.grade_map[x]
        elif field == 'subGrade':
            return lambda x: self.subgrade_map[x]
        elif field in ['isIncV', 'isIncVJoint']:
            return lambda x: self.income_verification[x]
        elif field == 'initialListStatus':
            return lambda x: self.init_status_dict[x]
        elif field == 'empLength':
            return self._convert_empLength
        elif field == 'addrZip':
            return lambda x: int(x[:3])
        else:
            return lambda :np.nan 

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

