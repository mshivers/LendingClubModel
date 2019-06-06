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
    previous_ira_investment = defaultdict(lambda :0)
    previous_tax_investment = defaultdict(lambda :0)
    employers = dict() 
    
    def __init__(self, ira_account, tax_account):
        self.ira_account = ira_account 
        self.tax_account = tax_account
        self.ira_cash = self.ira_account.get_cash_balance()
        self.tax_cash = self.tax_account.get_cash_balance()
        self.ira_summary = self.ira_account.get_account_summary()
        self.tax_summary = self.tax_account.get_account_summary()
        ira_notes_owned = self.ira_account.get_notes_owned()
        tax_notes_owned = self.tax_account.get_notes_owned()
        self.update_notes_owned(ira_notes_owned, account='ira') 
        self.update_notes_owned(tax_notes_owned, account='tax') 
        self.load_employers()
        #print(json.dumps(self.ira_summary, indent=4))

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
        loan['ira_invested_amount'] = cls.previous_ira_investment[loan.id]
        loan['tax_invested_amount'] = cls.previous_tax_investment[loan.id]
        if loan.id in cls.employers.keys():
            loan['currentCompany'] = cls.employers[loan.id]

    @classmethod
    def is_new(cls, loan):
        return loan['id'] not in cls.loans.keys()

    def get_cash_balance(self, account):
        if account=='ira':
            return self.ira_summary['availableCash']
        elif account=='tax':
            return self.tax_summary['availableCash']
        else:
            return 0

    def get_account_value(self, account):
        if account=='ira':
            return self.ira_summary['accountTotal']
        elif account=='tax':
            return self.tax_summary['accountTotal']
        else:
            return 0

    def get_past_due_adjustment(self, account):
        adjustment = 0
        if account=='ira':
            adjustment = self.ira_summary['adjustments']['adjustmentForPastDueNotes']
        elif account=='tax':
            adjustment = self.tax_summary['adjustments']['adjustmentForPastDueNotes']
        return adjustment 
        
    def get_adjusted_value(self, account):
        return self.get_account_value(account) - self.get_past_due_adjustment(account)
        
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
        return [loan for loan in self.all_loans() if loan['staged_ira_amount']>0 or loan['staged_tax_amount']>0]

    def loans_to_stage(self, account):
        '''returns a set of loan ids to stage'''
        to_stage = list() 
        for id, loan in self.loans.items():
            amount = self.stage_amount(loan, account)
            elapsed = (dt.now()-loan.init_time).total_seconds()
            if amount > 0:
                to_stage.append(loan)
        return to_stage

    def stage_amount(self, loan, account='ira'):
        if account == 'ira':
            return max(0, loan['max_ira_investment'] - loan['staged_ira_amount'] - self.previous_ira_investment[loan.id])
        elif account == 'tax':
            return max(0, loan['max_tax_investment'] - loan['staged_tax_amount'] - self.previous_tax_investment[loan.id])
        return 0

    def update_notes_owned(self, notes, account='ira'):
        if account == 'ira':
            for note in notes:
                self.previous_ira_investment[note['loanId']] += note['noteAmount']
        elif account == 'tax':
            for note in notes:
                self.previous_tax_investment[note['loanId']] += note['noteAmount']

    def report(self):
        recent_loans = self.recent_loans()
        if len(recent_loans) > 0:
            df = pd.DataFrame(recent_loans)
            df['is_ira_staged'] = (df['staged_ira_amount'] > 0).astype(int)
            df['is_tax_staged'] = (df['staged_tax_amount'] > 0).astype(int)
            grade_grp = df.groupby('subGradeString')
            series = {'numFound':grade_grp['id'].count()}
            series['loanAmount'] = grade_grp['loanAmount'].sum()
            series['maxIRR'] = grade_grp['irr'].max()
            series['minIRA'] = grade_grp['required_ira_return'].min()
            series['numIRA'] = grade_grp['is_ira_staged'].sum()
            series['maxIRRTax'] = grade_grp['irr_after_tax'].max()
            series['minTax'] = grade_grp['required_tax_return'].min()
            series['numTax'] = grade_grp['is_tax_staged'].sum()
            order = ['numFound', 'loanAmount', 'maxIRR', 'minIRA', 
            'numIRA', 'maxIRRTax', 'minTax', 'numTax']

            info_by_grade = pd.DataFrame(series).loc[:, order]
            formatters = dict()
            formatters['loanAmount'] = lambda x: '${:0,.0f}'.format(x)
            #formatters['meanIRR'] = lambda x: '{:1.2f}%'.format(100*x)
            formatters['maxIRRTax'] = lambda x: '{:1.2f}%'.format(100*x)
            formatters['maxIRR'] = lambda x: '{:1.2f}%'.format(100*x)
            formatters['minIRA'] = lambda x: 'N/A' if np.isnan(x) else '{:1.2f}%'.format(100*x)
            formatters['minTax'] = lambda x: 'N/A' if np.isnan(x) else '{:1.2f}%'.format(100*x)

            recent_loan_value = df['loanAmount'].sum()
            msg_list = list()
            num_staged = len(self.staged_loans())
            value_ira_staged = df['staged_ira_amount'].sum()
            value_tax_staged = df['staged_tax_amount'].sum()
            num_ira_staged = (df['staged_ira_amount']>0).sum()
            num_tax_staged = (df['staged_tax_amount']>0).sum()

            header = 'IRA Cash: ${:1,.0f};'.format(self.get_cash_balance('ira'))
            header += ' Value: ${:1,.0f};'.format(self.get_account_value('ira'))
            header += ' AdjValue: ${:1,.0f}'.format(self.get_adjusted_value('ira'))
            header += ' (${:1,.0f})\n'.format(self.get_past_due_adjustment('ira'))

            header += 'Tax Cash: ${:1,.0f};'.format(self.get_cash_balance('tax'))
            header += ' Value: ${:1,.0f};'.format(self.get_account_value('tax'))
            header += ' AdjValue: ${:1,.0f}'.format(self.get_adjusted_value('tax'))
            header += ' (${:1,.0f})\n'.format(self.get_past_due_adjustment('tax'))

            header += '{} total loans found, valued at ${:1,.0f}\n'.format(len(recent_loans), recent_loan_value)
            header += '{} IRA orders have been staged, totaling ${:1,.0f}.\n'.format(num_ira_staged, value_ira_staged)
            header += '{} Tax orders have been staged, totaling ${:1,.0f}.'.format(num_tax_staged, value_tax_staged)
            msg_list.append(header)

            msg_list.append(info_by_grade.to_string(formatters=formatters, col_space=7))

            for loan in self.staged_loans():
                msg_list.append(loan.detail_str())
            msg_list.append('https://www.lendingclub.com/auth/login')
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
            print('Waiting {} seconds'.format(wait))
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
    min_return = dict()
    min_return['tax'] = 0.025
    min_return['ira'] = 0.06
    participation_pct = 0.20
    learning_rate = 2

    def __init__(self, ira_cash=0, tax_cash=0):
        self.ira_cash = ira_cash
        self.tax_cash = tax_cash
        self.load_required_returns()

    def __del__(self):
        fname = paths.get_file('required_returns')
        json.dump(self.required_return, open(fname, 'w'), indent=4, sort_keys=True)

    def compute_current_ira_allocation(self, notes):
        notes = pd.DataFrame(notes)
        alloc = notes.groupby('grade')['principalPending'].sum()
        alloc = alloc / (self.ira_cash + alloc.sum())
        alloc_dict = defaultdict(lambda :0, alloc.to_dict())
        self.current_allocation = alloc_dict

    def load_required_returns(self):
        fname = paths.get_file('required_returns')
        self.required_return = json.load(open(fname, 'r'))
        for account, returns in self.required_return.items():
            for grade, val in returns.items(): 
                self.required_return[account][grade] = max(val, self.min_return[account])
        json.dump(self.required_return, open(fname + '.old', 'w'), indent=4, sort_keys=True)

    def raise_required_return(self, grade, account):
        adj = (1.0 / self.participation_pct) - 1
        self.required_return[account][grade] += adj * self.learning_rate * self.one_bps

    def lower_required_return(self, grade, account):
        self.required_return[account][grade] -= self.learning_rate * self.one_bps
        if self.required_return[account][grade] < self.min_return[account]:
            self.required_return[account][grade] = self.min_return[account]

    def get_required_return(self, grade, account):
        if grade in self.required_return[account].keys():
            return self.required_return[account][grade]
        else:
            return np.nan

    def set_max_investment(self, loan):
        max_ira_invest_amount = 0
        grade = loan['subGradeString']
        loan['required_ira_return'] = self.get_required_return(grade, 'ira')
        if loan['required_ira_return'] is not np.nan:
            if loan['irr'] > loan['required_ira_return']:
                excess_yield_in_bps = max(0, loan['irr'] - loan['required_ira_return']) / self.one_bps
                max_ira_invest_amount =  300 + min(500, 2*excess_yield_in_bps)
                self.raise_required_return(grade, 'ira')
            else:
                self.lower_required_return(grade, 'ira')
        if self.ira_cash < 10000:
            max_ira_invest_amount *= 0.5
        if self.ira_cash < 3000:
            max_ira_invest_amount *= 0.5
        max_ira_investment = 25 * np.floor(max_ira_invest_amount / 25)
        loan['max_ira_investment'] = max_ira_investment

        loan['required_tax_return'] = self.get_required_return(grade, 'tax')
        max_tax_investment = 0
        if loan['required_tax_return'] is not np.nan:
            if loan['irr_after_tax'] > loan['required_tax_return']:
                self.raise_required_return(grade, 'tax')
                if self.tax_cash > 10000:
                    max_tax_investment = 200 
                elif self.tax_cash > 3000:
                    max_tax_investment = 100 
                else:
                    max_tax_investment = 50 
            else:
                self.lower_required_return(grade, 'tax')
        loan['max_tax_investment'] = max_tax_investment


class PortfolioManager(object):
    def __init__(self, model_dir=None, new_only=True):
        self.ira_account = lclib.LendingClub('ira')
        self.tax_account = lclib.LendingClub('tax')
        if model_dir is None:
            model_dir = paths.get_dir('training')
        self.model_dir = model_dir
        self.quant = Quant(self.model_dir)
        self.backoffice = BackOffice(self.ira_account, self.tax_account)
        self.allocator = Allocator(ira_cash=self.backoffice.get_cash_balance('ira'),
                                   tax_cash=self.backoffice.get_cash_balance('tax'))

        self.employer = EmployerName('hjg')        
        if new_only:
            self.mark_old_loans()

    def mark_old_loans(self):
        loans = self.ira_account.get_listed_loans(new_only=False)
        print('{}: Found {} old listed loans.'.format(dt.now(), len(loans)))
        for loan_data in loans:
            loan = Loan(loan_data)
            loan['is_new'] = False
            self.quant.validate(loan)
            #    self.quant.run_models(loan)
            self.backoffice.track(loan)
 
    def search_for_yield(self):
        ira_staged = 0
        tax_staged = 0
        loans = self.ira_account.get_listed_loans(new_only=True)
        new_loans = [loan for loan in loans if self.backoffice.is_new(loan)] 
        print('{}: Found {} listed loans; {} new loans.'.format(dt.now(), len(loans), len(new_loans)))
        for loan_data in sorted(new_loans, key=lambda x:x['intRate'], reverse=True):
            loan = Loan(loan_data)
            if self.quant.validate(loan):
                self.quant.run_models(loan)
                self.allocator.set_max_investment(loan)                    
                self.backoffice.track(loan)
                ira_staged += self.maybe_submit_ira_order(loan)    
                tax_staged += self.maybe_submit_tax_order(loan)    
                loan.print_description() 
        '''
        if ira_staged:
            self.ira_account.submit_staged_orders()
        if tax_staged:
            self.tax_account.submit_staged_orders()
        '''

    def maybe_submit_ira_order(self, loan):
        amount_staged = 0
        amount_to_stage = self.backoffice.stage_amount(loan, 'ira')
        if amount_to_stage > 0:
            amount_staged = self.ira_account.submit_new_order(loan.id, amount_to_stage)
            loan['staged_ira_amount'] += amount_staged
            loan['ira_invested_amount'] += amount_staged
            if amount_staged > 0: 
                print('Submitted ${} for loan {} for {}'.format(amount_staged, loan.id, loan['empTitle']))
            else:
                print('Attempted to submit ${} for {}... FAILED'.format(amount_to_stage, loan['empTitle']))
        return amount_staged
         
    def maybe_submit_tax_order(self, loan):
        amount_staged = 0
        amount_to_stage = self.backoffice.stage_amount(loan, 'tax')
        if amount_to_stage > 0:
            amount_staged = self.tax_account.submit_new_order(loan.id, amount_to_stage)
            loan['staged_tax_amount'] += amount_staged
            loan['tax_invested_amount'] += amount_staged
            if amount_staged > 0: 
                print('Submitted ${} for loan {} for {}'.format(amount_staged, loan.id, loan['empTitle']))
            else:
                print('Attempted to submit ${} for {}... FAILED'.format(amount_to_stage, loan['empTitle']))
        return amount_staged
          
    def maybe_stage_ira_order(self, loan):
        amount_staged = 0
        amount_to_stage = self.backoffice.stage_amount(loan, 'ira')
        if amount_to_stage > 0:
            amount_staged = self.ira_account.stage_new_order(loan.id, amount_to_stage)
            loan['staged_ira_amount'] += amount_staged
            if amount_staged > 0: 
                print('IRA: staged ${} for loan {} for {}'.format(amount_staged, loan.id, loan['empTitle']))
            else:
                print('IRA: Attempted to Restage ${} for {}... FAILED'.format(amount_to_stage, loan['empTitle']))
        return amount_staged

    def maybe_stage_tax_order(self, loan):
        amount_staged = 0
        amount_to_stage = self.backoffice.stage_amount(loan, 'tax')
        if amount_to_stage > 0:
            amount_staged = self.tax_account.stage_new_order(loan.id, amount_to_stage)
            loan['staged_tax_amount'] += amount_staged
            if amount_staged > 0: 
                print('Tax: staged ${} for loan {} for {}'.format(amount_staged, loan.id, loan['empTitle']))
            else:
                print('Tax: Attempted to Restage ${} for {}... FAILED'.format(amount_to_stage, loan['empTitle']))
        return amount_staged

    def attempt_to_restage_ira_loans(self):
        loans_to_stage = self.backoffice.loans_to_stage(account='ira')
        staged = 0
        if loans_to_stage:
            print('\n\nFound {} loans to restage'.format(len(loans_to_stage)))
            for loan in loans_to_stage: 
                staged += self.maybe_stage_ira_order(loan)
            for loan in loans_to_stage:
                loan.print_description() 
        if staged:
            self.ira_account.submit_staged_orders()
    
    def check_employer(self, loan):
        if loan['currentCompany'] is None:
            loan['currentCompany'] = self.employer.get(loan.id)
            print(dt.now(), self.employer.name_count, self.employer.auth_count,)
            print(loan['id'], loan['subGradeString'], )
            if loan['irr'] is not None:
                print('{:1.2f}%'.format(100*loan['irr']), )
            print(loan['empTitle'], loan['currentCompany'])

    def get_staged_employers(self):
        staged_loans = self.backoffice.staged_loans()
        for loan in staged_loans:
            self.check_employer(loan)

    def get_remaining_employers(self):
        missing_employer_loans = [loan for loan in self.backoffice.all_loans() if loan['currentCompany'] is None]
        print('{} loans without employers found'.format(len(missing_employer_loans)))
        for loan in missing_employer_loans:
            if self.employer.auth_count <= 10:
                self.check_employer(loan)

    def try_for_awhile(self, minutes=10):
        start = dt.now()
        end = start + td(minutes=minutes)
        while True: 
            self.search_for_yield()
            #self.get_staged_employers()
            print(self.backoffice.report())
            if dt.now() > end:
                break
            down_time = min(10, utils.sleep_seconds())
            print('Napping for {} seconds'.format(down_time))
            sleep(down_time)
            print('\n\nTry Again...\n\n')
            #self.attempt_to_restage_ira_loans()
        print('Done trying!')
        emaillib.send_email(self.backoffice.report()) 
        #self.get_remaining_employers()


class Loan(dict):
    def __init__(self, features):
        self.update({ 'max_ira_investment': 0,
                      'max_tax_investment': 0,
                      'staged_ira_amount': 0,
                      'staged_tax_amount': 0,
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
        print(self.detail_str())

    def detail_str(self):
        loan = self
        pstr = ''
        if (loan['ira_invested_amount'] or loan['staged_ira_amount'] or
            loan['tax_invested_amount'] or loan['staged_tax_amount']):
            istr = list()
            if loan['ira_invested_amount']:
                istr.append('IRA Invested: ${}'.format(loan['ira_invested_amount']))
            if loan['staged_ira_amount']:
                istr.append('IRA Staged: ${}'.format(loan['staged_ira_amount']))
            if loan['tax_invested_amount']:
                istr.append('Tax Invested: ${}'.format(loan['tax_invested_amount']))
            if loan['staged_tax_amount']:
                istr.append('Tax Staged: ${}'.format(loan['staged_tax_amount']))
            pstr += ' | '.join(istr)
            pstr += '\n'

        pstr += 'IRR: {:1.2f}%'.format(100*loan['irr'])
        pstr += ' | IRRTax: {:1.2f}%'.format(100*loan['irr_after_tax'])
        if loan['required_ira_return']:
            pstr += ' | MinIRR: {:1.2f}%'.format(100*loan['required_ira_return'])
        if loan['required_tax_return']:
            pstr += ' | MinTax: {:1.2f}%'.format(100*loan['required_tax_return'])
        pstr += ' | IntRate: {}%'.format(loan['intRate'])
        pstr += ' | Default: {:1.2f}%'.format(100*loan['default_risk'])
        pstr += ' | Prepay: {:1.2f}%'.format(100*loan['prepay_risk'])

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



