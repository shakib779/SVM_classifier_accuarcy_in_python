import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit

dataset = pd.read_csv('train.txt')
dataset.drop(['id', 'date'], 1, inplace = True)

X = np.array(dataset.drop(['Occupancy'], 1))
y = np.array(dataset['Occupancy'])

kf = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1, random_state = 0)

sum = 0.0

for train_index, test_index in kf.split(X, y):
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	model = svm.SVC(gamma = 'auto')
	model.fit(X_train, y_train)
	accuracy = model.score(X_test, y_test)
	sum += accuracy
	#print(accuracy)

accuracy = sum/(kf.get_n_splits(X, y)*1.0);
print('Accuracy of Model = %s' %accuracy)