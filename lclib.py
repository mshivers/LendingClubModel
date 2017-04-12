from datalib import ReferenceData
import constants
from personalized import p
from session import Session
import requests
import utils 

class LendingClub(object):

    def __init__(self, account):
        ''' Valid accounts are 'ira' or 'tax' as defined in personalized.py '''
        self.account_type = account
        self.email = p.get_email(account)
        self.id = p.get_id(account)
        self.key = p.get_key(account)
        self._pass = p.get_pass(account)
        self.session = Session(email=self.email, password=self._pass) 
        self.session.authenticate()
            
    def get_listed_loans(self, new_only=True):
        loans = []
        try:
            result = requests.get('https://api.lendingclub.com/api/investor/v1/loans/listing', 
                                  headers={'Authorization': self.key,
                                           'X-LC-LISTING-VERSION': 1.1},
                                  params={'showAll': not new_only})
            if result.status_code == 200:  #success
                result_js = result.json()
                if 'loans' in result_js.keys():
                    loans = result.json()['loans']
        except:
            pass
        return loans

    def get_cash_balance(self):
        cash = -1
        url = 'https://api.lendingclub.com/api/investor/v1/accounts/{}/availablecash'.format(self.id)
        try:
            result = requests.get(url, headers={'Authorization': self.key})
            if result.status_code == 200:  #success
                result_js = result.json()
                if 'availableCash' in result_js.keys():
                    cash = result.json()['availableCash']
        except:
            pass
        return cash 
    
    def get_notes_owned(self):
        notes = []
        url = 'https://api.lendingclub.com/api/investor/v1/accounts/{}/detailednotes'.format(self.id)
        result = requests.get(url, headers={'Authorization': self.key})
        if result.status_code == 200:  #success
            result_js = result.json()
            if 'myNotes' in result_js.keys():
                notes = result.json()['myNotes']
        return notes


    def stage_order(self, loan_id, amount):
        amount_staged = 0
        payload = {
            'method': 'addToPortfolio',
            'loan_id': loan_id,
            'loan_amount': int(amount),
            'remove': 'false'
        }
        try:
            response = self.session.get('/data/portfolio', query=payload)
            json_response = response.json()
        except:
            log.write('{}: Failed prestage orders\n'.format(dt.now()))
            print '\nFailed to prestage order {}\n'.format(loan_id)
            return 0

        if json_response['result']=='success':
            if 'add_modifications' in json_response.keys():
                mod = json_response['add_modifications']
                if 'loanFractions' in mod.keys():
                    frac = mod['loanFractions']
                    if isinstance(frac, list):
                        frac = frac[0]
                    if isinstance(frac, dict) and 'loanFractionAmountAdded' in frac.keys():
                        amount_staged =  frac['loanFractionAmountAdded'] 
            else:
                amount_staged = amount
        return amount_staged 

    def get_loan_details(self, loan_id):
        '''
        Returns the loan details, including location, current job title, 
        employer, relisted status, and number of inquiries.
        '''
        payload = {
            'loan_id': loan_id
        }
        try:
            response = self.session.post('/browse/loanDetailAj.action', data=payload)
            detail = response.json()
            return detail 
        except:
            return -1

    def get_employer_name(self, loan_id):
        currentCompany = None 
        result = self.get_loan_details(loan_id)
        if isinstance(result, dict):
            currentCompany = utils.only_ascii(result['currentCompany'])
        return currentCompany



class APIDataParser(object):
    ''' 
    Manages the parsing of the Lending Club API data:
    1.  Converts string fields to numerical values, consistent with the historical data
    2.  Fills NaN values with numeric defaults
    3.  Converts numeric fields to make sure they are in the same units as the historical data
    '''
    reference_data = ReferenceData()
    def __init__(self):
        self.api_fields = self.reference_data.get_loanstats2api_map().values()
        self.string_converter = constants.StringToConst()
        self.ok_to_be_null = ['dtiJoint',
                              'desc',
                              'isIncVJoint',
                              'investorCount',
                              'annualIncJoint',
                              'housingPayment',
                              'mtgPayment'
                              ]

    def null_fill_value(self, field):
        if( field.startswith('mthsSinceLast')
                or field.startswith('mthsSinceRecent')
                or field.startswith('moSinRcnt')
                or field.startswith('mthsSinceRcnt')):
            return constants.LARGE_INT
        elif (field.startswith('moSinOld')
                or field.startswith('num')
                or field.endswith('Util')
                or field == 'percentBcGt75'
                or field == 'bcOpenToBuy'
                or field == 'empLength'):
            return constants.NEGATIVE_INT 
        elif field=='empTitle':
            return ''
        else:
            return None
    
    def null_fill_fields(self):
        return [f for f in self.api_fields if self.null_fill_value(f) is not None]

    def parse(self, data):
        valid = True 
        if 'valid' in data.keys():
            if data['valid'] == True:
                return valid 

        for k in self.api_fields:
            if k not in data.keys():
                print 'Field {} is missing'.format(k)
                valid = False 

        for k,v in data.items():
            if v is None:
                data[k] = self.null_fill_value(k) 
                
            if type(v) in [str, unicode]:
                if 'String' not in k:
                    data[k] = self.string_converter.convert(k, v)                
                    if data[k] != v:
                        data[u'{}String'.format(k)] = v

            if data[k] is None and k not in self.ok_to_be_null:
                print 'Field {} has a null value; check api_parser defaults'.format(k)
                valid = False

        #API empLength is given in months. Convert to years
        if data['empLength'] not in range(-1, 11):
            data['empLength'] = min(11, data['empLength'] / 12)

        data['empTitle'] = utils.format_title(data['empTitle'])
        data['valid'] = valid 
        return valid


def calc_model_sensitivities(loans):
    for loan in loans:
        loancopy = copy.deepcopy(loan)
        loancopy['clean_title_log_odds'] = 0
        loan['dflt_ctlo_zero'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['capitalization_log_odds'] = 0
        loan['dflt_caplo_zero'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['HPA1Yr'] = -5.0 
        loan['dflt_hpa1y_neg5pct'] = model.calc_default_risk(loancopy)        
        loancopy['HPA1Yr'] = 0 
        loan['dflt_hpa1y_zero'] = model.calc_default_risk(loancopy)        
        loancopy['HPA1Yr'] = 5.0
        loan['dflt_hpa1y_5pct'] = model.calc_default_risk(loancopy)        
        loancopy['HPA1Yr'] = 10.0
        loan['dflt_hpa1y_10pct'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['avg_urate'] = 0.03
        loancopy['urate'] = 0.03 
        loan['dflt_urate_3pct'] = model.calc_default_risk(loancopy)        
        loancopy['avg_urate'] = 0.08
        loancopy['urate'] = 0.08 
        loan['dflt_urate_8pct'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['urate_chg'] = 0.02
        loan['dflt_urate_chg_2pct'] = model.calc_default_risk(loancopy)        
        loancopy['urate_chg'] = -0.02
        loan['dflt_urate_chg_neg2pct'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['RENT']
        loan['dflt_rent'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['MORTGAGE']
        loan['dflt_mortgage'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['home_status_number'] = lclib.home_map['OWN']
        loan['dflt_own'] = model.calc_default_risk(loancopy)        

        loancopy = copy.deepcopy(loan)
        loancopy['even_loan_amount'] = int(not loan['even_loan_amount']) 
        loan['dflt_not_even_loan_amount'] = model.calc_default_risk(loancopy)        

    return



def update_recent_loan_info(known_loans, info):
    irr_list = list()
    irr_by_grade = defaultdict(lambda :list())
    irr_data = list()
    for l in known_loans.values():
        if l['base_irr'] > -100:
            elapsed = (dt.now()-l['search_time']).total_seconds()
            if elapsed < 600:
                irr_list.append(l['base_irr']) 
                irr_by_grade[l['grade']].append(l['base_irr'])
                irr_data.append((l['grade'], l['initialListStatus'], l['base_irr']))

    info['average_irr'] = np.mean(irr_list)
    info['irr_by_grade'] = irr_by_grade

    col_names = ['grade', 'initialListStatus', 'base_irr']
    if len(irr_data)==0:
        irr_data = [('A', 'F', -100)]
    info['irr_df'] = pd.DataFrame(data=irr_data, columns=col_names)
    return

def get_loans_to_save(known_loans):
    loans_to_save = list()
    for l in known_loans.values():
        if l['inputs_parsed']==True and l['details_saved']==False and l['currentCompany']!='':
            loans_to_save.append(l)
    return loans_to_save

def get_recently_staged(known_loans):
    recently_staged = list()
    for l in known_loans.values():
        if l['staged_amount'] > 0:
            if 'staged_time' in l.keys():
                if (dt.now()-l['staged_time']).total_seconds() < 3600:
                    recently_staged.append(l)
    return sort_by_int_rate(recently_staged)

def attempt_to_stage(lc, known_loans):
    #Try to stage or restage any loan that didn't get a full allocation
    staged_loans = list()
    for id, l in known_loans.items():
        elapsed = (dt.now() - l['search_time']).total_seconds()
        if (l['staged_amount']<l['max_stage_amount']) and (elapsed < 3600):
            amount_to_invest = l['max_stage_amount'] - l['staged_amount']
            amount_staged = stage_order_fast(lc, l['id'], amount_to_invest)
            if amount_staged > 0:
                l['staged_amount'] += amount_staged
                l['staged_time'] = dt.now()
                cash -= amount_staged 
                staged_loans.append(l)
                print 'Restaged ${} for loan {} for {}'.format(amount_staged, l['id'], l['emp_title']) 
            else:
                print 'Attempted to Restage ${} for {}... FAILED'.format(amount_to_invest, l['emp_title'])

    return staged_loans



