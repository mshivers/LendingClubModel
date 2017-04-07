
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

 
class Inventory(object):
    def __init__(self):
        self.loans = dict()

    def add(self, loan):
        self.loans[loan.id] = loan
        
    def id_exists(self, id):
        return id in self.loans.keys()




class Investor(object):
    def __init__(self, model_dir=None):
        self.model_dir = model_dir
        self.account = LendingClub('ira')
        self.feature_manager = lclib.FeatureManager(model_dir)
        self.loans = Inventory()

    def check_for_new_loans(self):
        listed_loans = self.account.get_listed_loans(new_only=True)
        for loan_data in listed_loans:
            if not self.loans.id_exists(loan_data['id']):
               self.feature_manager.process(loan_data)
               self.loans.add(Loan(loan_data))



