import pandas as pd
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn import metrics

file_path_test = os.getcwd() + '/adult.test'
file_path_train = os.getcwd() + '/adult.data'

adult_train = pd.read_csv(file_path_train,header=None)
adult_test = pd.read_csv(file_path_test,header=None)

# features_to_encode_train = adult_train.columns[adult_train.dtypes==object].tolist()
# print(features_to_encode_train)
# col_trans = make_column_transformer((OneHotEncoder(),features_to_encode_train),remainder='passthrough')
#rf_classifier = RandomForestClassifier(n_estimators=100)
# clf = make_pipeline(col_trans, rf_classifier)
# print(clf)

# features_to_encode_test = adult_test.columns[adult_test.dtypes==object].tolist()
# print(features_to_encode_test)
# col_trans_test = make_column_transformer((OneHotEncoder(),features_to_encode_test),remainder='passthrough')
# rf_classifier_test = RandomForestClassifier(n_estimators=100)
# clf = make_pipeline(col_trans_test, rf_classifier_test)
# print(clf)

y_test = adult_test[14]
x_test = adult_test[adult_test.columns[:-1]]
y_train = adult_train[14]
x_train = adult_train[adult_train.columns[:-1]]

features_to_encode = x_train.columns[x_train.dtypes==object].tolist()
print(features_to_encode)
col_trans = make_column_transformer((OneHotEncoder(),features_to_encode),remainder='passthrough')
rf_classifier = RandomForestClassifier(n_estimators=100)
clf = make_pipeline(col_trans, rf_classifier)

clf.fit(x_test,y_test)
metrics.RocCurveDisplay.from_estimator(clf,x_test,y_test)
plt.savefig('roc_curve.PNG')
print(metrics.roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))