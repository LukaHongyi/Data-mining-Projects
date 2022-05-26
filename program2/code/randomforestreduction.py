import imp
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from dataprocess import lowvariance, randomforests, splitdata
import matplotlib.pyplot as plt

dataMY = pd.read_csv("MYdata.data", header=None, delimiter=',')
dataBO = pd.read_csv("BOdata.data", header=None, delimiter=',')
dataHK = pd.read_csv("HKdata.data", header=None, delimiter=',')
dataMY = randomforests(dataMY)
dataBO = randomforests(dataBO)
dataHK = randomforests(dataHK)
xtrain_files1, ytrain_files1, xtest_files1, ytest_files1, X_test1, Y_test1 = splitdata(dataMY, 0.1)
xtrain_files2, ytrain_files2, xtest_files2, ytest_files2, X_test2, Y_test2 = splitdata(dataBO, 0.1)
xtrain_files3, ytrain_files3, xtest_files3, ytest_files3, X_test3, Y_test3 = splitdata(dataHK, 0.1)

metricarrs=[[1,"euclidean"],[1,"manhattan"],[1,"minkowski"],
[2,"euclidean"],[2,"manhattan"],[2,"minkowski"],
[3,"euclidean"],[3,"manhattan"],[3,"minkowski"],
[4,"euclidean"],[4,"manhattan"],[4,"minkowski"],
[5,"euclidean"],[5,"manhattan"],[5,"minkowski"]]
metricarrsplot=["1 and euclidean","1 and manhattan","1 and minkowski",
"2 and euclidean","2 and manhattan","2 and minkowski",
"3 and euclidean","3 and manhattan","3 and minkowski",
"4 and euclidean","4 and manhattan","4 and minkowski",
"5 and euclidean","5 and manhattan","5 and minkowski"]
myaccy=[]
boaccy=[]
hkaccy=[]
for metricarr in metricarrs:
    neigh1 = KNeighborsClassifier(n_neighbors=metricarr[0],metric=metricarr[1])
    neigh2 = KNeighborsClassifier(n_neighbors=metricarr[0],metric=metricarr[1])
    neigh3 = KNeighborsClassifier(n_neighbors=metricarr[0],metric=metricarr[1])
    acc1, acc2, acc3=0, 0, 0
    for i in range(5):
        neigh1.fit(xtrain_files1[i], ytrain_files1[i])
        test_predictions1 = neigh1.predict(xtest_files1[i])
        acc1 += accuracy_score(ytest_files1[i], test_predictions1)
        neigh2.fit(xtrain_files2[i], ytrain_files2[i])
        test_predictions2 = neigh2.predict(xtest_files2[i])
        acc2 += accuracy_score(ytest_files2[i], test_predictions2)
        neigh3.fit(xtrain_files3[i], ytrain_files3[i])
        test_predictions3 = neigh3.predict(xtest_files3[i])
        acc3 += accuracy_score(ytest_files3[i], test_predictions3)
        # print(accuracy_score(ytest_files[i], test_predictions))
    acc1 = acc1/5
    myaccy.append(acc1)
    acc2 = acc2/5
    boaccy.append(acc2)
    acc3 = acc3/5
    hkaccy.append(acc3)

plt.xticks(fontsize=6)
plt.plot(metricarrsplot,myaccy, c='red', marker='s', label='M and Y')
plt.plot(metricarrsplot,boaccy, c='g', marker='o', label='B and O')
plt.plot(metricarrsplot,hkaccy, c='black', marker='x', label='H and K')
plt.legend()
plt.show()