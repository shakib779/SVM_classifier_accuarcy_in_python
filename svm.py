import numpy as np
from sklearn import svm
import pandas as pd


train = pd.read_csv('train.txt')
test = pd.read_csv('test.txt')
train.drop(['id', 'date'], 1, inplace = True)
test.drop(['id', 'date'], 1, inplace = True)

print(train)
print(test)

X_train = np.array(train.drop(['Occupancy'], 1))
y_train = np.array(train['Occupancy'])

print(X_train)
print(y_train)

X_test = np.array(test.drop(['Occupancy'], 1))
y_test = np.array(test['Occupancy'])

print(X_test)
print(y_test)

model = svm.SVC()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(accuracy)