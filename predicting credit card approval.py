import numpy as np
import pandas as pd
cc_apps = pd.read_csv('cc_approvals.data')
cc_apps.head()
cc_apps = pd.read_csv('cc_approvals.data')
cc_apps.head()
cc_apps_description = cc_apps.describe()
print(cc_apps_description)
cc_apps_info = cc_apps.info()
print(cc_apps_info)
cc_apps.tail(17)
from sklearn.model_selection import train_test_split
cc_apps = cc_apps.drop([11, 13], axis=0)
cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)
cc_apps_train = cc_apps_train.replace('?',np.NaN)
cc_apps_test = cc_apps_test.replace('?',np.NaN)
cc_apps_train.fillna(cc_apps_train.mean(), inplace=True)
cc_apps_test.fillna(cc_apps_train.mean(), inplace=True)
print(cc_apps_train.isnull().sum())
print(cc_apps_test.isnull().sum())
for col in cc_apps_train.columns:
   if cc_apps_train[col].dtypes == 'object':
        cc_apps_train = cc_apps_train.fillna(cc_apps_train[col].value_counts().index[0])
        cc_apps_test = cc_apps_test.fillna(cc_apps_train[col].value_counts().index[0])
print(cc_apps_train.isnull().sum())
print(cc_apps_test.isnull().sum())
cc_apps_train = pd.get_dummies(cc_apps_train)
cc_apps_test = pd.get_dummies(cc_apps_test)
cc_apps_test = cc_apps_test.reindex(columns=cc_apps_train.columns, fill_value=0)
from sklearn.preprocessing import MinMaxScaler
X_train, y_train = cc_apps_train.iloc[:, :-1].values, cc_apps_train.iloc[:, [-1]].values
X_test, y_test = cc_apps_test.iloc[:, :-1].values, cc_apps_test.iloc[:, [-1]].values
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(rescaledX_train,y_train)
from sklearn.metrics import confusion_matrix
y_pred = logreg.predict(rescaledX_test)
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test,y_test))
confusion_matrix(y_test,y_pred)
from sklearn.model_selection import GridSearchCV
tol = [0.01, 0.001 ,0.0001]
max_iter = [100, 150, 200]
param_grid = dict(tol=tol, max_iter=max_iter)
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
grid_model_result = grid_model.fit(rescaledX_train, y_train)
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
best_model = grid_model_result.best_estimator_
print("Accuracy of logistic regression classifier: ", best_model.score(rescaledX_test,y_test))