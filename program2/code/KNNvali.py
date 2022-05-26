import imp
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from dataprocess import lowvariance, randomforests, splitdata
import matplotlib.pyplot as plt
import time
dataMY = df = pd.read_csv("MYdata.data", header=None, delimiter=',')
dataBO = df = pd.read_csv("BOdata.data", header=None, delimiter=',')
dataHK = df = pd.read_csv("HKdata.data", header=None, delimiter=',')
dataMY = randomforests(dataMY)
dataBO = randomforests(dataBO)
dataHK = randomforests(dataHK)
X1 = dataMY.iloc[:,1:]
Y1 = dataMY.iloc[:,0]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.1, random_state=0)
X2 = dataBO.iloc[:,1:]
Y2 = dataBO.iloc[:,0]
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.1, random_state=0)
X3 = dataHK.iloc[:,1:]
Y3 = dataHK.iloc[:,0]
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y3, test_size=0.1, random_state=0)


start =time.perf_counter()
neigh1 = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
neigh2 = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
neigh3 = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
acc1, acc2, acc3=0, 0, 0
neigh1.fit(X_train1, Y_train1)
test_predictions1 = neigh1.predict(X_test1)
acc1 = accuracy_score(Y_test1, test_predictions1)
neigh2.fit(X_train2, Y_train2)
test_predictions2 = neigh2.predict(X_test2)
acc2 = accuracy_score(Y_test2, test_predictions2)
neigh3.fit(X_train3, Y_train3)
test_predictions3 = neigh3.predict(X_test3)
acc3 = accuracy_score(Y_test3, test_predictions3)
end = time.perf_counter()
print('Running time: %s Seconds'%(end-start), "M and Y: %f"%(acc1), "B and O: %f"%(acc2), "H and K: %f"%(acc3))