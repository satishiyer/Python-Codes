import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

cc = pd.read_csv("C:\\Users\\Satish Iyer\\Downloads\\creditcard.csv")
cc
cc.drop(columns='Unnamed: 0',inplace=True)
cc

cc.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cc[['income']],cc.card,test_size=0.1)
X_test
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_train)
model.score(X_train,y_train)
model.predict_proba(X_train)

import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

sal = pd.read_csv("C:\\Users\\Satish Iyer\\Downloads\\sal.csv")
sal
sal.describe()
#as python require numeric
# Dropping the columns contains string
sal.drop(["workclass","education","maritalstatus","occupation","relationship","race","sex","native"],inplace=True,axis = 1)

salary["cat"] = 0
salary.loc[salary.Salary==" <=50K","cat"] = 1
salary.Salary.value_counts()
salary.cat.value_counts()
