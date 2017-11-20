from datalib import ReferenceData
import constants
from personalized import p
from session import Session
import requests
import json
import utils 
import numpy as np
from time import sleep
from bs4 import BeautifulSoup

class LendingClub(object):

    def __init__(self, account):
        ''' Valid accounts are 'ira' or 'tax' as defined in personalized.py '''
        self.account_type = account
        self.email = p.get_email(account)
        self.id = p.get_id(account)
        self.key = p.get_key(account)
        self._pass = p.get_pass(account)
        self.session = Session(email=self.email, password=self._pass) 
        self.attempt_to_authenticate()
            
    def attempt_to_authenticate(self):
        count = 1
        success = False
        while (success==False and count < 5):
            print('{} login attempt #{}'.format(self.account_type, count))
            try:
                success = self.session.authenticate()
                print('Login succeeded')
            except:
                print('Login Failed')
                sleep(3) 
            count += 1

    def get_listed_loans(self, new_only=True):
        loans = []
        result = requests.get('https://api.lendingclub.com/api/investor/v1/loans/listing', 
                              headers={'Authorization': self.key,
                                       'X-LC-LISTING-VERSION': '1.1'},
                              params={'showAll': not new_only})
        if result.status_code == 200:  #success
            result_js = result.json()
            if 'loans' in result_js.keys():
                loans = result.json()['loans']
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

    def get_account_summary(self):
        summary = {}
        url = 'https://api.lendingclub.com/api/investor/v1/accounts/{}/summary'.format(self.id)
        try:
            result = requests.get(url, headers={'Authorization': self.key})
            if result.status_code == 200:  #success
                summary = result.json()
        except:
            pass
        return summary
    
    def get_notes_owned(self):
        notes = []
        url = 'https://api.lendingclub.com/api/investor/v1/accounts/{}/detailednotes'.format(self.id)
        result = requests.get(url, headers={'Authorization': self.key})
        if result.status_code == 200:  #success
            result_js = result.json()
            if 'myNotes' in result_js.keys():
                notes = result.json()['myNotes']
        return notes

    def submit_new_order(self, loan_id, amount):
        invested_amount = 0
        url = 'https://api.lendingclub.com/api/investor/v1/accounts/{}/orders'.format(self.id) 
        headers={'Authorization': self.key, 'Content-Type': 'application/json'}
        payload={'aid':self.id,
                'orders':[
                  {
                      'loanId':loan_id,
                      'requestedAmount':amount
                   }]
               }
                              
        result = requests.post(url, headers=headers, data=json.dumps(payload))
        if result.status_code == 200:  #success
            result_js = result.json()
            if 'orderConfirmations' in result_js.keys():
                invested_amount = result_js['orderConfirmations'][0]['investedAmount']
        return invested_amount


    def stage_new_order(self, loan_id, amount):
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
            print('\nFailed to prestage order {}\n'.format(loan_id))
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

    def get_strut_token(self):
        """
        Move the staged loan notes to the order stage and get the struts token
        from the place order HTML.
        The order will not be placed until calling _confirm_order()
        Returns
        -------
        dict
            A dict with the token name and value
        """

        try:
            # Move to the place order page and get the struts token

            response = self.session.get('/portfolio/placeOrder.action')
            soup = BeautifulSoup(response.text, "html5lib")


            # Example HTML with the stuts token:
            """
            <input type="hidden" name="struts.token.name" value="token" />
            <input type="hidden" name="token" value="C4MJZP39Q86KDX8KN8SBTVCP0WSFBXEL" />
            """
            # 'struts.token.name' defines the field name with the token value

            strut_tag = None
            strut_token_name = soup.find('input', {'name': 'struts.token.name'})
            if strut_token_name and strut_token_name['value'].strip():

                # Get form around the strut.token.name element
                form = soup.form # assumed
                for parent in strut_token_name.parents:
                    if parent and parent.name == 'form':
                        form = parent
                        break

                # Get strut token value
                strut_token_name = strut_token_name['value']
                strut_tag = soup.find('input', {'name': strut_token_name})
                if strut_tag and strut_tag['value'].strip():
                    return {'name': strut_token_name, 'value': strut_tag['value'].strip()}


        except Exception as e:
            return {'name': '', 'value': ''}

    def submit_staged_orders(self):
        """
        Returns
        -------
        int
            The completed order ID.
        """
        order_id = 0
        response = None

        token = self.get_strut_token()

        if not token or token['value'] == '':
            return

        # Process order confirmation page
        try:
            # Place the order
            payload = {}
            if token:
                payload['struts.token.name'] = token['name']
                payload[token['name']] = token['value']
            response = self.session.post('/portfolio/orderConfirmed.action', data=payload)

            # Process HTML for the order ID
            html = response.text
            soup = BeautifulSoup(html, 'html5lib')

            # Order num
            order_field = soup.find(id='order_id')
            if order_field:
                order_id = int(order_field['value'])

            return order_id
        except:
            pass



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
        self.ok_to_be_null = [
                              'dtiJoint',
                              'desc',
                              'isIncVJoint',
                              'investorCount',
                              'annualIncJoint',
                              'housingPayment',
                              'mtgPayment',
                              'reviewStatusD'
                              ]
        self.required_nonzero = [
                                 'annualInc', 
                                 'loanAmount',
                                 'totHiCredLim',
                                 'totCurBal'
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
                print('Field {} is missing'.format(k))
                valid = False 


        for k in list(data.keys()):
            v = data[k]
            if v is None:
                data[k] = self.null_fill_value(k) 
                
            if type(v) == str:
                if 'String' not in k:
                    data[k] = self.string_converter.convert(k, v)                
                    if data[k] != v:
                        data[u'{}String'.format(k)] = v

            if data[k] is None and k not in self.ok_to_be_null:
                import pdb
                pdb.set_trace()
                print('Field {} has a null value; check api_parser defaults'.format(k))
                valid = False
       
        for k in self.required_nonzero:   #these fields are denominators of simple features
            if data[k] <= 0:
                valid = False

        # use joint info where both exist:
        if data['dtiJoint'] is not None:
            data['dti'] = data['dtiJoint']
        if data['annualIncJoint'] is not None:
            data['annualInc'] = data['annualIncJoint']

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


