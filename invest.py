import lclib
import features
import curves
from constants import paths
from datetime import datetime as dt

class BackOffice(object):
    loans = dict()

    def __init__(self):
        pass

    @classmethod
    def track(cls, loan):
        cls.loans[loan.id] = loan 
 
    @classmethod
    def is_new(cls, loan_data):
        return loan_data['id'] not in cls.loans.keys()


class Quant(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.api_data_parser = lclib.APIDataParser()
        self.feature_manager = features.FeatureManager(model_dir)
        self.default_curves = curves.DefaultCurve(self.model_dir)
        self.prepay_curves = curves.PrepayCurve(self.model_dir)

    def validate(self, loan_data):
        return self.api_data_parser.parse(loan_data)

    def run_models(self, loan_data):
        self.feature_manager.process(loan_data)

        



class PortfolioManager(object):
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = paths.get_dir('training')
        self.model_dir = model_dir
        self.quant = Quant(self.model_dir)
        self.account = lclib.LendingClub('ira')
        self.backoffice = BackOffice()

    def search_for_yield(self):
        loans = self.account.get_listed_loans(new_only=True)
        for loan_data in loans:
            if self.backoffice.is_new(loan_data):
                if self.quant.validate(loan_data):
                    self.quant.run_models(loan_data)
                    self.backoffice.track(Loan(loan_data))
         



class Loan(object):
    def __init__(self, features):
        self.search_time = dt.now()
        self.id = features['id']
        self.features = features 
        self.model_outputs = dict()
        self.invested = {
                        'max_stage_amount': 0,
                        'staged_amount': 0
                        }
        self.flags = {
                      'details_saved': False,
                      'email_details': False
                      }

    def id(self):
        return self.id

    def get(self, feature):
        if feature in self.features.keys():
            return self.features[feature]
        else:
            return None

    def detail_str(self):
        pstr = ''
        #pstr += 'BaseIRR: {:1.2f}%'.format(100*loan['base_irr'])
        #pstr += ' | StressIRR: {:1.2f}%'.format(100*loan['stress_irr'])
        #pstr += ' | BaseIRRTax: {:1.2f}%'.format(100*loan['base_irr_tax'])
        #pstr += ' | StressIRRTax: {:1.2f}%'.format(100*loan['stress_irr_tax'])
        pstr += ' | IntRate: {}%'.format(loan['intRate'])

        pstr += '\nDefaultRisk: {:1.2f}%'.format(100*loan['default_risk'])
        pstr += ' | PrepayRisk: {:1.2f}%'.format(100*loan['prepay_risk'])
        #pstr += ' | RiskFactor: {:1.2f}'.format(loan['risk_factor'])

        pstr += '\nInitStatus: {}'.format(loan['initialListStatus'])
        #pstr += ' | Staged: ${}'.format(loan['staged_amount'])

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
      
        pstr += '\nJobTitle: {}'.format(loan['empTitle'])
        #pstr += ' | Company: {}'.format(loan['currentCompany'])
        
        pstr += '\nClean Title Log Odds: {:1.2f}'.format(loan['clean_title_log_odds'])
        pstr += ' | Capitalization Log Odds: {:1.2f}'.format(loan['capitalization_log_odds'])
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


