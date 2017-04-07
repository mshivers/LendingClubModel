import lclib
import datetime
t0 = datetime.datetime.now()
#lclib.get_external_data()
#lclib.update_all_training_data()
#lclib.estimate_default_curves()
#lclib.estimate_prepay_curves()

import build_default_rf
import build_prepay_rf

t1 = datetime.datetime.now()
print 'Elapsed: {}'.format((t1-t0).total_seconds())
