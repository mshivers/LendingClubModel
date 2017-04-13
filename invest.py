import lclib
import features
import curves
import utils
import numpy as np
from constants import paths
from datetime import datetime as dt
from datetime import timedelta as td 
from time import sleep
from collections import defaultdict

class BackOffice(object):
    loans = dict()
    invested = defaultdict(lambda :0)

    def __init__(self, notes_owned):
        self.update_notes_owned(notes_owned) 

    @classmethod
    def track(cls, loan):
        cls.loans[loan.id] = loan 
 
    @classmethod
    def is_new(cls, loan):
        return loan.id not in cls.loans.keys()

    def get(self, id):
        if id in self.loans.keys():
            return self.loans[id]
        else:
            return None

    def all_loans(self):
        return self.loans.values()

    def loans_to_stage(self):
        '''returns a set of loan ids to stage'''
        to_stage = list() 
        for id, loan in self.loans.items():
            amount = self.stage_amount(loan)
            elapsed = (dt.now()-loan.init_time).total_seconds()
            if amount > 0 and elapsed < 30 * 60:
                to_stage.append(loan)
        return to_stage

    def staged_loans(self):
        return [loan for loan in self.all_loans() if loan['staged_amount']>0]

    def stage_amount(self, loan):
        return max(0, loan['max_investment'] - loan['staged_amount'] - self.invested[loan.id])

    def update_notes_owned(self, notes):
        self.invested.update([(note['loanId'], note['loanAmount']) for note in notes])


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
    def __init__(self):
        self.account = lclib.LendingClub('hjg')
        self.name_count = 0
        self.auth_count = 0
        
    def get(self, id):
        if not self.account.session.is_site_available():
            self.account.session.authenticate()
            self.auth_count += 1
        name = self.account.get_employer_name(id)
        self.name_count += 1
        return name


class PortfolioManager(object):
    def __init__(self, model_dir=None, required_return=0.08):
        if model_dir is None:
            model_dir = paths.get_dir('training')
        self.model_dir = model_dir
        self.required_return = required_return
        self.quant = Quant(self.model_dir)
        self.account = lclib.LendingClub('ira')
        self.backoffice = BackOffice(self.account.get_notes_owned())
        self.employer = EmployerName()        

    def search_for_yield(self):
        loans = self.account.get_listed_loans(new_only=True)
        for loan_data in sorted(loans, key=lambda x:x['intRate'], reverse=True):
            loan = Loan(loan_data)
            if self.backoffice.is_new(loan):
                if self.quant.validate(loan):
                    self.quant.run_models(loan)
                    self.set_max_investment(loan)                    
                    self.backoffice.track(loan)
                    self.maybe_stage_loan(loan)    
                    loan.print_description() 
         
    def set_max_investment(self, loan):
        if loan['irr'] > self.required_return:
            if loan['gradeString'] < 'G':
                excess_yield_dollars = 100
                excess_yield = 100 * max(0, (loan['irr'] - self.required_return))
                max_invest_amount = excess_yield_dollars * excess_yield 
                max_invest_amount = 25 * np.ceil(max_invest_amount / 25)
                loan['max_investment'] = max_invest_amount 

    def maybe_stage_loan(self, loan):
            amount_to_stage = self.backoffice.stage_amount(loan)
            if amount_to_stage > 0:
                amount_staged = self.account.stage_order(loan.id, amount_to_stage)
                loan['staged_amount'] += amount_staged
                if amount_staged > 0: 
                    print 'staged ${} for loan {} for {}'.format(amount_staged, loan.id, loan['empTitle']) 
                else:
                    print 'Attempted to Restage ${} for {}... FAILED'.format(amount_to_stage, loan['empTitle'])
                if loan['currentCompany'] is None:
                    loan['currentCompany'] = self.employer.get(loan.id)
                print dt.now(), self.employer.name_count, self.employer.auth_count
       
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
            print dt.now(), self.employer.name_count, self.employer.auth_count

    def summarize_staged(self):
        staged_loans = self.backoffice.staged_loans()
        if staged_loans:
            print 'Summary of {} staged loans:'.format(len(staged_loans))
            for loan in staged_loans:
                self.check_employer(loan)
                loan.print_description()

    def try_for_awhile(self, minutes=10, min_irr=0.09):
        start = dt.now()
        end = start + td(minutes=minutes)
        while dt.now() < end:
            self.search_for_yield()
            self.summarize_staged()
            down_time = min(10, utils.sleep_seconds())
            print 'Napping for {} seconds'.format(down_time)
            sleep(down_time)
            print 'Try Again...',
            self.attempt_to_restage_loans()
        print 'Done trying!'
        

class Loan(dict):
    def __init__(self, features):
        self.update({ 'max_investment': 0,
                      'staged_amount': 0,
                      'details_saved': False,
                      'email_details': False
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
        pstr += ' | IntRate: {}%'.format(loan['intRate'])
        pstr += ' | DefaultRisk: {:1.2f}%'.format(100*loan['default_risk'])
        pstr += ' | PrepayRisk: {:1.2f}%'.format(100*loan['prepay_risk'])
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



'''
1. get new loans from LendingClub()
2. parse the raw api data to update base features
3. add zip3 features
4. add binary features
5. add model features
6. add default and prepayment curves
7. calc IRR
8. determine how much to invest
9. stage loans
10. send email
11. save data
'''


