import numpy as np
import pandas as pd
# NECESSARY IMPORTS

import numpy as np 
import pandas as pd 
import pickle



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.model_selection import cross_val_score




import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def wage_person_convertor(value):
    
    if value == '<=50K':
        return 0
    else:
        return 1

def good_job(occupation):
    if occupation in ['Prof-specialty','Exec-managerial','Armed-Forces']:
        return 2
    elif occupation in ['Adm-clerical','Sales']:
        return 1
    else:
        return 0
    
            


train = pd.read_csv('train.csv', header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',  'marital_status', 'occupation','relationship', 'race', 
'sex', 'capital_gain',  'capital_loss', 'hours_per_week', 'native_country', 'wage_class']
train.columns = col_labels

train['wage_class']=train['wage_class'].astype('object')
train['wage_class_number'] = train.apply(lambda x :wage_person_convertor(x['wage_class']),axis=1)

train['good_job_number'] = train.apply(lambda x :good_job(x['occupation']),axis=1)
train['sex_number'] = train.apply(lambda x :1 if x['sex']=='Male' else 0,axis=1)

X= train[['age','education_num','good_job_number','sex_number']]
Y= train['wage_class_number']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#model = LogisticRegression()
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())


# Dumping the data
pickle.dump(model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
