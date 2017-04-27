import lclib
import features
import curves
import utils
import numpy as np
import emaillib
from constants import paths
from datetime import datetime as dt
from datetime import timedelta as td 
from time import sleep
from collections import defaultdict

class BackOffice(object):
    loans = dict()
    invested = defaultdict(lambda :0)
    employers = defaultdict(lambda :None)
    
    def __init__(self, notes_owned=None):
        self.update_notes_owned(notes_owned) 
        self.load_employers()

    def __del__(self):
        with open(paths.get_file('employer_data'), 'a') as f:
            for loan in self.all_loans():
                if loan['currentCompany'] is not None:
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
        if loan['currentCompany'] is None:
            loan['currentCompany'] = cls.employers[loan.id]
            print 'Loaded employers data: {}: {}'.format(loan.id, loan['currentCompany']) 

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

    def recent_loans(self):
        return [loan for loan in self.loans if loan.init_time.hour == dt.now().hour]

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
        info = pd.DataFrame([(l['loanAmount'], l['grade'], l['IRR'], l['staged_amount']) for l in recent_loans],
                columns = ['loanAmount', 'grade', 'IRR', 'staged'])

        recent_loan_value = info['loanAmount'].sum()

        g = info.groupby('grade')
        def compute_metrics(x):
            result = {'irr_count': x['IRR'].count(), 'irr_mean': x['IRR'].mean()}
            return pd.Series(result, name='metrics')
        recent_info_by_grade = g.apply(compute_metrics)
        bought_info_by_grade = info.ix[info.staged>0].groupby('grade').apply(compute_metrics)
   
        irr_msgs = list()
        irr_msgs.append('Average IRR is {:1.2f}%.'.format(100*info['average_irr']))
        for grade in sorted(info['irr_by_grade'].keys()):
            avg = 100*np.mean(info['irr_by_grade'][grade])
            num = len(info['irr_by_grade'][grade])
            bought = int(count_by_grade[grade])
            irr_msgs.append('Average of {} grade {} IRRs is {:1.2f}%; {} staged.'.format(num, grade, avg, bought))

        msg_list = list()
        msg_list.append('{} orders have been staged.'.format(len(self.staged_loans())))
        msg_list.append('{} total loans found, valued at ${:1,.0f}'.format(len(recent_loans), recent_loan_value))
        msg_list.append(irr_by_grade.to_string())
        msg_list.append('\r\n'.join(irr_msgs))
        for loan in self.staged_loans():
            msg_list.append(loan.detail_str())
        msg_list.append('https://www.lendingclub.com/account/gotoLogin.action')
        msg_list.append('Send at MacOSX clocktime {}'.format(dt.now()))
        msg = '\r\n\n'.join(msg_list) 
        send_email(msg)
        return


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
        self.account = lclib.LendingClub(account_name)
        self.name_count = 0
        self.auth_count = 0
        self.next_call = dt.now()
      
    def sleep_seconds(self):
        return 5
 
    def seconds_to_wake(self):
        return max(0, (self.next_call - dt.now()).total_seconds())

    def set_next_call_time(self):
        self.next_call = dt.now() + td(seconds=self.sleep_seconds())
 
    def get(self, id):
        self.name_count += 1
        if dt.now() < self.next_call:
            wait = self.seconds_to_wake()
            print 'Waiting {} seconds'.format(wait)
            sleep(wait)
        name = self.account.get_employer_name(id)
        self.set_next_call_time()
        if name is None:
            sleep(10)
            try:
                self.account.session.authenticate()
            except:
                pass
            self.auth_count += 1
            name = self.account.get_employer_name(id)
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
        self.employer = EmployerName('hjg')        

    def search_for_yield(self):
        loans = self.account.get_listed_loans(new_only=True)
        print '{}: Found {} listed loans.'.format(dt.now(), len(loans))
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
                excess_yield_in_bps = 10000 * max(0, loan['irr'] - self.required_return)
                max_invest_amount = 100 + min(150, excess_yield_in_bps)
                loan['max_investment'] = 25 * np.floor(max_invest_amount / 25)

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
            print loan['empTitle'], loan['currentCompany']

    def summarize_staged(self):
        staged_loans = self.backoffice.staged_loans()
        msg = list() 
        msg.append('Summary of {} staged loans:'.format(len(staged_loans)))
        if staged_loans:
            for loan in staged_loans:
                self.check_employer(loan)
                msg.append(loan.detail_str())
        return '\n'.join(msg)

    def get_remaining_employers(self):
        for loan in self.backoffice.all_loans():
            self.check_employer(loan)

    def try_for_awhile(self, minutes=10, min_irr=0.09):
        start = dt.now()
        end = start + td(minutes=minutes)
        while dt.now() < end:
            self.search_for_yield()
            print self.summarize_staged()
            down_time = min(10, utils.sleep_seconds())
            print 'Napping for {} seconds'.format(down_time)
            sleep(down_time)
            print '\n\nTry Again...\n\n'
            self.attempt_to_restage_loans()
        print 'Done trying!'
        msg = self.summarize_staged()
        emaillib.send_email(msg) 


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
        pstr += ' | Default: {:1.2f}%'.format(100*loan['default_risk'])
        pstr += ' | Prepay: {:1.2f}%'.format(100*loan['prepay_risk'])
        pstr += ' | Invested: ${}'.format(loan['invested_amount'])
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



