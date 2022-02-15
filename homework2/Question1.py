"""
CSCE 587 Homework 2 Question 1
Created on February 9, 2022
@Author: James Carroll
"""

import pandas as pd
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import sklearn.metrics as mt
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import sklearn.tree as tr

#Get file path for fetal_health.csv
file_path = os.getcwd() + '/fetal_health.csv'

#Part A
#Take in csv and get train and test sets, and then split these sets into X and Y
#Fit the x and y train sets and predict the test set lables
df = pd.read_csv(file_path)
df_train = df.iloc[:1500]
df_test = df.iloc[1500:]
y_test = df_test.fetal_health
x_test = df_test[df_test.columns[:-1]]
y_train = df_train.fetal_health
x_train = df_train[df_train.columns[:-1]]
clf = DecisionTreeClassifier()
clf.fit(x_test, y_test)
pred = clf.predict(x_test)
print('Prediction-------------------------------------------------------')
print(pred)

#Part B
#Create a confusion matrix and get the tp, tn, fp, and fn
cm = confusion_matrix(y_test, pred)
print('\nConfusion Matrix-----------------------------------------------')
print(cm)
print('\nConfusion Matrix Items-----------------------------------------')
tp = cm[0][0]
print('TP', tp)
fn = cm[1][0] + cm[2][0]
print('FN', fn)
fp = cm[0][1] + cm[0][2]
print('FP', fp)
tn = cm[1][1] + cm[2][1] + cm[1][2] + cm [2][2]
print('TN', tn)

#Part C
#Get recall, precision, f1, and mcc scores
ps = mt.precision_score(y_test, pred, average=None)
print('\nScores-----------------------------------------------------------')
print('Precison Score: ', ps)
recall = mt.recall_score(y_test, pred, average=None)
print('Recall Score: ', recall)
f1 = mt.f1_score(y_test, pred, average=None)
print('F1 Score: ', f1)
mcc = mt.matthews_corrcoef(y_test, pred)
print('MCC Score: ', mcc)

#Part D
#Create a bar chart and decision tree for the training data, these are saved to PNG files for easy viewing
#Decision tree is also printed in a ttext format
print('\nPlot and Decision Tree saved as PNGs included in zip file')
plt.style.use('ggplot')
model = DecisionTreeRegressor()
model.fit(x_train, y_train)
importance = model.feature_importances_
plt.bar([x for x in range(len(importance))], importance, color='green')
plt.savefig("Plot.png")
fig = plt.figure(figsize=(25,20))
_ = tr.plot_tree(model)
print(tr.plot_tree(model))
fig.savefig("DecisionTree.PNG")