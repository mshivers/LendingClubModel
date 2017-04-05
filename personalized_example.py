

class MyData(object):
    parent_dir = '/Users/myLoginName/LCModel'

    _tax_email = 'your.email@gmail.com'
    _tax_id = 888888888 
    _tax_key = 'yourLongRandomString2'

    _ira_email = 'your.other.email@gmail.com'
    _ira_id = 999999999 
    _ira_key = 'yourLongRandomString2'

    _smtp_email = 'your.email@gmail.com'
    _smtp_key = 'yoursmtpkey'  

    @classmethod
    def get_id(cls, account):
        if account=='tax':
            return cls._tax_id
        elif account=='ira':
            return cls._ira_id
        else:
            raise Exception('Invalid account')

    @classmethod
    def get_email(cls, account):
        if account=='tax':
            return cls._tax_email
        elif account=='ira':
            return cls._ira_email
        elif account=='smtp':
            return cls._smtp_email
        else:
            raise Exception('Invalid account')

    @classmethod
    def get_key(cls, account):
        if account=='tax':
            return cls._tax_key
        elif account=='ira':
            return cls._ira_key
        elif account=='smtp':
            return cls._smtp_key
        else:
            raise Exception('Invalid account')


p = MyData()
