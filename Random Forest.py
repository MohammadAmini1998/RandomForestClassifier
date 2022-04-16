
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os 
import warnings
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score
import graphviz

from IPython.display import Image


warnings.filterwarnings('ignore')

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df=pd.read_csv("car_evaluation.csv",header=None,names=col_names)

print(df.head(20))
#here we see if there is a missing data or not ... ! 
print(df.isnull().sum())

X=df[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
Y=df[['class']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)


encoder = ce.OrdinalEncoder(cols=['buying', 'maint',
                                 'doors', 'persons',
                                  'lug_boot', 'safety'])
X_train=encoder.fit_transform(X_train)
X_test=encoder.transform(X_test)


rfc=RandomForestClassifier(n_estimators=1000,random_state=10)



rfc.fit(X_train,Y_train)

y_pred=rfc.predict(X_test)

print('Model accuracy score with 10 decision-trees : {0:0.4f}'.format(accuracy_score(Y_test, y_pred)))


#feature importance 
feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feature_scores) # here we find that doors are not that important


