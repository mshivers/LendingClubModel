import lclib
import features
import curves
import utils
import json
import numpy as np
import pandas as pd
import emaillib
from constants import paths
from datetime import datetime as dt
from datetime import timedelta as td 
from time import sleep
from collections import defaultdict

class BackOffice(object):
    loans = dict()
    invested = defaultdict(lambda :0)
    employers = dict() 
    
    def __init__(self, notes_owned=None):
        self.update_notes_owned(notes_owned) 
        self.load_employers()

    def __del__(self):
        with open(paths.get_file('employer_data'), 'a') as f:
            for loan in self.all_loans():
                if loan['id'] not in self.employers.keys():
                    if loan['currentCompany'] is not None:
                        if (loan['currentCompany'] != 'n/a') or (loan['empTitle'] == '^$'):
                            f.write('{}|{}\n'.format(loan.id, loan['currentCompany']))

    @classmethod
    def load_employers(cls):
        employer_data = open(paths.get_file('employer_data'), 'r').read().split('\n')
        employer_names = [row.split('|') for row in employer_data if '|' in row]
        employer_names = [(int(row[0]), row[1]) for row in employer_names] 
        cls.employers.update(employer_names)

    @classmethod
    def track(cls, loan):
        cls.loans[loan.id] = loan 
        loan['invested_amount'] = cls.invested[loan.id]
        if loan.id in cls.employers.keys():
            loan['currentCompany'] = cls.employers[loan.id]

    @classmethod
    def is_new(cls, loan):
        return loan['id'] not in cls.loans.keys()

    def get(self, _id):
        if _id in self.loans.keys():
            return self.loans[_id]
        else:
            return None

    def all_loans(self):
        return self.loans.values()

    def recent_loans(self):
        return [loan for loan in self.loans.values() if loan['is_new'] == True]

    def staged_loans(self):
        return [loan for loan in self.all_loans() if loan['staged_amount']>0]

    def loans_to_stage(self):
        '''returns a set of loan ids to stage'''
        to_stage = list() 
        for id, loan in self.loans.items():
            amount = self.stage_amount(loan)
            elapsed = (dt.now()-loan.init_time).total_seconds()
            if amount > 0 and elapsed < 30 * 60:
                to_stage.append(loan)
        return to_stage

    def stage_amount(self, loan):
        return max(0, loan['max_investment'] - loan['staged_amount'] - self.invested[loan.id])

    def update_notes_owned(self, notes):
        for note in notes:
            self.invested[note['loanId']] += note['noteAmount']

    def report(self):
        recent_loans = self.recent_loans()
        if len(recent_loans) > 0:
            df = pd.DataFrame(recent_loans)
            df['is_staged'] = (df['staged_amount'] > 0).astype(int)
            grade_grp = df.groupby('subGradeString')
            series = {'numFound':grade_grp['id'].count()}
            series['loanAmount'] = grade_grp['loanAmount'].sum()
            series['meanIRR'] = grade_grp['irr'].mean()
            series['maxIRR'] = grade_grp['irr'].max()
            series['maxIRRTax'] = grade_grp['irr_after_tax'].max()
            series['numStaged'] = grade_grp['is_staged'].sum()
            series['reqReturn'] = grade_grp['required_return'].mean()

            info_by_grade = pd.DataFrame(series)
            formatters = dict()
            formatters['loanAmount'] = lambda x: '${:0,.0f}'.format(x)
            formatters['meanIRR'] = lambda x: '{:1.2f}%'.format(100*x)
            formatters['maxIRRTax'] = lambda x: '{:1.2f}%'.format(100*x)
            formatters['maxIRR'] = lambda x: '{:1.2f}%'.format(100*x)
            formatters['reqReturn'] = lambda x: '{:1.2f}%'.format(100*x)

            recent_loan_value = df['loanAmount'].sum()
            msg_list = list()
            msg_list.append('{} orders have been staged.'.format(len(self.staged_loans())))
            msg_list.append('{} total loans found, valued at ${:1,.0f}'.format(len(recent_loans), recent_loan_value))

            msg_list.append(info_by_grade.to_string(formatters=formatters, col_space=10))

            for loan in self.staged_loans():
                msg_list.append(loan.detail_str())
            msg_list.append('https://www.lendingclub.com/')
            msg_list.append('Send at MacOSX clocktime {}'.format(dt.now()))
            msg = '\r\n\n'.join(msg_list) 
        else:
            msg = 'No loans found\n\n'
        return msg


class Quant(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.api_data_parser = lclib.APIDataParser()
        self.feature_manager = features.FeatureManager(model_dir)
        self.cashflow_model = curves.CashFlowModel(model_dir)

    def validate(self, loan_data):
        return self.api_data_parser.parse(loan_data)

    def run_models(self, loan_data):
        self.feature_manager.process_lookup_features(loan_data)
        self.feature_manager.process_forest_features(loan_data)
        self.cashflow_model.process(loan_data) 
        

class EmployerName(object):
    def __init__(self, account_name=None):
        if account_name is None:
            account_name = 'hjg'
        self.account_name = account_name
        self.account = None 
        self.name_count = 0
        self.auth_count = 0
        self.next_call = dt.now()
      
    def sleep_seconds(self):
        return 6
 
    def seconds_to_wake(self):
        return max(0, (self.next_call - dt.now()).total_seconds())

    def set_next_call_time(self):
        self.next_call = dt.now() + td(seconds=self.sleep_seconds())
 
    def get(self, id):
        if self.account is None:
            self.account = lclib.LendingClub(self.account_name)
        self.name_count += 1
        if dt.now() < self.next_call:
            wait = self.seconds_to_wake()
            print 'Waiting {} seconds'.format(wait)
            sleep(wait)
        name = self.account.get_employer_name(id)
        self.set_next_call_time()
        if name is None:
            sleep(2)
            self.account.session.get(self.account.session.base_url)
            self.auth_count += 1
            sleep(2)
            name = self.account.get_employer_name(id)
            self.set_next_call_time()
        return name

class Allocator(object):
    one_bps = 0.0001
    min_return = 0.08
    participation_pct = 0.20
    learning_rate = 4

    def __init__(self, notes, cash):
        self.compute_current_allocation(notes, cash)
        self.load_required_returns()

    def __del__(self):
        fname = paths.get_file('required_returns')
        json.dump(self.required_return, open(fname, 'w'), indent=4, sort_keys=True)

    def compute_current_allocation(self, notes, cash):
        notes = pd.DataFrame(notes)
        alloc = notes.groupby('grade')['principalPending'].sum()
        alloc = alloc / (cash + alloc.sum())
        alloc_dict = defaultdict(lambda :0, alloc.to_dict())
        self.current_allocation = alloc_dict

    def load_required_returns(self):
        fname = paths.get_file('required_returns')
        self.required_return = json.load(open(fname, 'r'))
        json.dump(self.required_return, open(fname + '.old', 'w'), indent=4, sort_keys=True)

    def raise_required_return(self, grade):
        adj = (1.0 / self.participation_pct) - 1
        self.required_return[grade] += adj * self.learning_rate * self.one_bps

    def lower_required_return(self, grade):
        self.required_return[grade] -= self.learning_rate * self.one_bps
        if self.required_return[grade] < self.min_return:
            self.required_return[grade] = self.min_return

    def get_required_return(self, grade):
        if grade in self.required_return.keys():
            return self.required_return[grade]
        else:
            return None

    def set_max_investment(self, loan):
        grade = loan['subGradeString']
        loan['required_return'] = self.get_required_return(grade)
        if loan['irr'] > loan['required_return']:
            excess_yield_in_bps = max(0, loan['irr'] - loan['required_return']) / self.one_bps
            max_invest_amount = 25 + min(75, excess_yield_in_bps)
            loan['max_investment'] = 25 * np.floor(max_invest_amount / 25)
            self.raise_required_return(grade)
        else:
            self.lower_required_return(grade)


class PortfolioManager(object):
    def __init__(self, model_dir=None, required_return=0.08):
        self.account = lclib.LendingClub('ira')
        if model_dir is None:
            model_dir = paths.get_dir('training')
        self.model_dir = model_dir
        self.quant = Quant(self.model_dir)
        self.cash = self.account.get_cash_balance()
        notes = self.account.get_notes_owned()
        self.backoffice = BackOffice(notes)
        self.allocator = Allocator(notes, self.cash)
        self.employer = EmployerName('hjg')        
        self.load_old_loans()

    def load_old_loans(self):
        loans = self.account.get_listed_loans(new_only=False)
        print '{}: Found {} old listed loans.'.format(dt.now(), len(loans))
        for loan_data in loans:
            loan = Loan(loan_data)
            loan['is_new'] = False
            self.quant.validate(loan)
            #    self.quant.run_models(loan)
            self.backoffice.track(loan)
 
    def search_for_yield(self):
        loans = self.account.get_listed_loans(new_only=True)
        new_loans = [loan for loan in loans if self.backoffice.is_new(loan)] 
        print '{}: Found {} listed loans; {} new loans.'.format(dt.now(), len(loans), len(new_loans))
        for loan_data in sorted(new_loans, key=lambda x:x['intRate'], reverse=True):
            loan = Loan(loan_data)
            if self.quant.validate(loan):
                self.quant.run_models(loan)
                self.allocator.set_max_investment(loan)                    
                self.backoffice.track(loan)
                self.maybe_stage_loan(loan)    
                loan.print_description() 
         
    def maybe_stage_loan(self, loan):
            amount_to_stage = self.backoffice.stage_amount(loan)
            if amount_to_stage > 0:
                amount_staged = self.account.stage_order(loan.id, amount_to_stage)
                loan['staged_amount'] += amount_staged
                if amount_staged > 0: 
                    print 'staged ${} for loan {} for {}'.format(amount_staged, loan.id, loan['empTitle']) 
                else:
                    print 'Attempted to Restage ${} for {}... FAILED'.format(amount_to_stage, loan['empTitle'])
       
    def attempt_to_restage_loans(self):
        loans_to_stage = self.backoffice.loans_to_stage()
        if loans_to_stage:
            print '\n\nFound {} loans to restage'.format(len(loans_to_stage))
            for loan in loans_to_stage: 
                self.maybe_stage_loan(loan)
            for loan in loans_to_stage:
                loan.print_description() 
    
    def check_employer(self, loan):
        if loan['currentCompany'] is None:
            loan['currentCompany'] = self.employer.get(loan.id)
            print dt.now(), self.employer.name_count, self.employer.auth_count,
            print loan['id'], loan['subGradeString'], 
            if loan['irr'] is not None:
                print '{:1.2f}%'.format(100*loan['irr']), 
            print loan['empTitle'], loan['currentCompany']

    def get_staged_employers(self):
        staged_loans = self.backoffice.staged_loans()
        for loan in staged_loans:
            self.check_employer(loan)

    def get_remaining_employers(self):
        missing_employer_loans = [loan for loan in self.backoffice.all_loans() if loan['currentCompany'] is None]
        print '{} loans without employers found'.format(len(missing_employer_loans))
        for loan in missing_employer_loans:
            if self.employer.auth_count <= 10:
                self.check_employer(loan)

    def try_for_awhile(self, minutes=10):
        start = dt.now()
        end = start + td(minutes=minutes)
        while True: 
            self.search_for_yield()
            self.get_staged_employers()
            print self.backoffice.report()
            if dt.now() > end:
                break
            down_time = min(10, utils.sleep_seconds())
            print 'Napping for {} seconds'.format(down_time)
            sleep(down_time)
            print '\n\nTry Again...\n\n'
            self.attempt_to_restage_loans()
        print 'Done trying!'
        emaillib.send_email(self.backoffice.report()) 
        self.get_remaining_employers()


class Loan(dict):
    def __init__(self, features):
        self.update({ 'max_investment': 0,
                      'staged_amount': 0,
                      'details_saved': False,
                      'email_details': False,
                      'is_new':True
                      })
        if isinstance(features, dict):
            self.update(features)
        self.id = self['id']
        self.init_time = dt.now()

    def __getitem__(self, x):
        if x in self.keys():
            return super(Loan, self).__getitem__(x)
        else:
            return None

    def print_description(self):
        print self.detail_str()

    def detail_str(self):
        loan = self
        pstr = ''
        pstr += 'IRR: {:1.2f}%'.format(100*loan['irr'])
        pstr += ' | IRRTax: {:1.2f}%'.format(100*loan['irr_after_tax'])
        if loan['required_return']:
            pstr += ' | MinIRR: {:1.2f}%'.format(100*loan['required_return'])
        pstr += ' | IntRate: {}%'.format(loan['intRate'])
        pstr += ' | Default: {:1.2f}%'.format(100*loan['default_risk'])
        pstr += ' | Prepay: {:1.2f}%'.format(100*loan['prepay_risk'])
        if loan['invested_amount']:
            pstr += ' | Invested: ${}'.format(loan['invested_amount'])
        if loan['staged_amount']:
            pstr += ' | Staged: ${}'.format(loan['staged_amount'])

        pstr += '\nLoanAmnt: ${:1,.0f}'.format(loan['loanAmount'])
        pstr += ' | Term: {}'.format(loan['term'])
        pstr += ' | Grade: {}'.format(loan['subGradeString'])
        pstr += ' | Purpose: {}'.format(loan['purposeString'])
        pstr += ' | LoanId: {}'.format(loan['id'])
        
        pstr += '\nRevBal: ${:1,.0f}'.format(loan['revolBal'])
        pstr += ' | RevUtil: {}%'.format(loan['revolUtil'])
        pstr += ' | DTI: {}%'.format(loan['dti'])
        pstr += ' | Inq6m: {}'.format(loan['inqLast6Mths'])
        pstr += ' | 1stCredit: {}'.format(loan['earliestCrLine'].split('T')[0])
        pstr += ' | fico: {}'.format(loan['ficoRangeLow'])
      
        pstr += '\nAccOpen24M: {}'.format(loan['accOpenPast24Mths'])
        pstr += ' | TradeLinesOpen12M: {}'.format(loan['numTlOpPast12m'])
        pstr += ' | MthsSinceOldestRevol: {}'.format(loan['moSinOldRevTlOp'])
        pstr += ' | IntPctIncome: {:1.0f}bps'.format(10000*loan['int_pct_inc'])
        pstr += ' | InitStatus: {}'.format(loan['initialListStatusString'])

        pstr += '\nJobTitle: {}'.format(loan['empTitle'])
        pstr += ' | Company: {}'.format(loan['currentCompany'])

        pstr += '\nDefault Odds: {:1.2f}'.format(loan['default_empTitle_shorttoks_odds'])
        pstr += ' | Prepay Odds: {:1.2f}'.format(loan['prepay_empTitle_shorttoks_odds'])
        pstr += ' | Income: ${:1,.0f}'.format(loan['annualInc'])
        pstr += ' | Tenure: {}'.format(loan['empLength'])

        pstr += '\nLoc: {},{}'.format(loan['addrZip'], loan['addrState'])
        pstr += ' | MedInc: ${:1,.0f}'.format(loan['census_median_income'])
        pstr += ' | URate: {:1.1f}%'.format(100*loan['urate'])
        pstr += ' | 12mChg: {:1.1f}%'.format(100*loan['urate_chg'])

        pstr += '\nHomeOwn: {}'.format(loan['homeOwnershipString'])
        pstr += ' | PrimaryCity: {}'.format(loan['primaryCity'])
        pstr += ' | HPA1: {:1.1f}%'.format(loan['hpa4'])
        pstr += ' | HPA5: {:1.1f}%'.format(loan['hpa20'])
        
        pstr += '\n\n'

        return pstr 



