import numpy as np
import pandas as pd
# import seaborn as sb
# import matplotlib.pyplot as plt
# import sklearn
import pickle 
# from pandas import Series, DataFrame
# from pylab import rcParams
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

Url = "https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
titanic = pd.read_csv(Url,usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Survived'])

# replacing null values with mean
titanic['Age'] = titanic['Age'].fillna(value=np.mean(titanic['Age']))

# converting Sex column 
titanic = pd.get_dummies(data=titanic,columns=['Sex'],drop_first=True)
titanic['Sex'] = titanic['Sex_male']
titanic.drop(axis=1,columns=['Sex_male'],inplace=True)

##------------------------------------Removiing outliers--------------------------
q = titanic['Age'].quantile(0.98)
# we are removing the top 2% data from the Age column
titanic = titanic[titanic['Age']<q]

q = titanic['SibSp'].quantile(0.99)
# we are removing the top 1% data from the SibSp column

titanic  = titanic[titanic['SibSp']<q]
q = titanic['Fare'].quantile(0.98)
# we are removing the top 2% data from the Fare column
titanic  = titanic[titanic['Fare']<q]
# -----------------------------------------------------------------------------------

# Test train spllit
X = titanic.drop(columns=['Survived'])
y = titanic['Survived']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.33,random_state=42)

# Hyperparameter tuning
model_tuned = DecisionTreeClassifier()
param_grid = {'criterion':["gini", "entropy", "log_loss"],
             "splitter":["best", "random"],
              'max_depth': [3, 6, 9],
              'max_leaf_nodes': [3, 6, 9],
              'max_features': ['sqrt', 'log2', None]
             }
clf = GridSearchCV(estimator=model_tuned,param_grid=param_grid,n_jobs=-1,cv=5,verbose=3)
clf.fit(x_train, y_train)

#printing best params and best estimator
print(clf.best_estimator_)
print(clf.best_params_)

## printing accuracy terms
Final_model = DecisionTreeClassifier(criterion='entropy',max_depth=6,max_features=None,max_leaf_nodes=9,splitter='best')
Final_model.fit(x_train,y_train)
y_pred_Final_model_test = Final_model.predict(x_test)
y_pred_Final_model_train = Final_model.predict(x_train)
print("For train dataset\n",classification_report(y_train,y_pred_Final_model_train),"\n\n")
print("For test dataset\n",classification_report(y_test,y_pred_Final_model_test))

## pickling the model
pickle.dump(Final_model,open("Dec_tree_cls_model2.pkl",'wb'))
