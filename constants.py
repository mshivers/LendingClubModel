import os
from collections import defaultdict
from personalized import p


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
    reference_data_dir = os.path.join(p.parent_dir, 'data/reference_data')
    bls_data_dir = os.path.join(p.parent_dir, 'data/bls_data')
    fhfa_data_dir = os.path.join(p.parent_dir, 'data/fhfa_data')
    saved_prod_data_dir = os.path.join(p.parent_dir, 'data/saved_prod_data')
    payments_file = os.path.join(loanstats_dir, 'PMTHIST_ALL_20170315.csv')
    cached_training_data_file = os.path.join(training_data_dir, 'cached_training_data.csv')
    base_data_file = os.path.join(training_data_dir, 'base_data.csv')

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
        else:
            return -1

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
                                'addrZip']
        self.home_map = dict([('ANY', 0), ('NONE',0), ('OTHER',0), 
                              ('RENT',1), ('MORTGAGE',2), ('OWN',3)])
        self.purpose_dict = defaultdict(lambda :np.nan)
        self.purpose_dict.update([('credit_card', 0), ('credit_card_refinancing', 0), 
                                  ('debt_consolidation',1), 
                                  ('home_improvement',2), 
                                  ('car',3), ('car_financing',3), 
                                  ('educational',4), 
                                  ('house',5), ('home_buying',5),
                                  ('major_purchase',6), 
                                  ('medical_expenses',7), ('medical',7), 
                                  ('moving',8), ('moving_and_relocation',8), 
                                  ('other',9),
                                  ('renewable_energy',10), ('green_loan',10),
                                  ('business',11),('small_business',11),
                                  ('vacation',12), 
                                  ('wedding',13)])
        grades = list('ABCDEFG')
        self.grade_map = defaultdict(lambda :np.nan, zip(grades, range(len(grades))))
        subgrades = ['{}{}'.format(l,n) for l in 'ABCDEFG' for n in range(1,6)]
        self.subgrade_map = defaultdict(lambda :np.nan, zip(subgrades, range(len(subgrades))))
        loanstats_verification_dict = dict([('Verified',2), ('Source Verified',1), ('Not Verified',0)]) 
        api_verification_dict = dict([('VERIFIED',2), ('SOURCE_VERIFIED',1), ('NOT_VERIFIED',0)])
        self.income_verification = loanstats_verification_dict
        self.income_verification.update(api_verification_dict)
        self.init_status_dict = dict([('f',0), ('F',0), ('w',1), ('W',1)])

    def _convert_empLength(self, value):
        value=value.replace('< 1 year', '0')
        value=value.replace('1 year','1')
        value=value.replace('10+ years', '11')
        value=value.replace('n/a', '-1')
        value=value.replace(' years', '')
        return int(value)
    
    def _convert_grade(self, value):
        return self.grade_map[value]

    def _convert_homeOwnership(self, value):
        return self.home_map[value.upper()]

    def _convert_purpose(self, value):
        value = value.lower().replace(' ', '_')
        return self.purpose_dict[value]

    def _convert_subGrade(self, value):
        return self.subgrade_map[value]

    def _convert_inc_verification(self, value):
        return self.income_verification[value]

    def _convert_initialListStatus(self, value):
        return self.init_status_dict[value]

    def _convert_addrZip(self, value):
        return int(value[:3])

    def convert(self, field, value):
        if field == 'homeOwnership':
            if value.upper() in self.home_map.keys():
                return self.home_map[value.upper()]
        elif field == 'purpose':
            value = value.lower().replace(' ', '_')
            if value in self.purpose_dict.keys():
                return self.purpose_dict[value]
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
            return lambda x: self.purpose_dict[x.lower().replace(' ', '_')]
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


