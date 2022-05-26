import imp
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity
from dataprocess import splitdata
import sklearn
import matplotlib.pyplot as plt

dataMY = df = pd.read_csv("MYdata.data", header=None, delimiter=',')
dataBO = df = pd.read_csv("BOdata.data", header=None, delimiter=',')
dataHK = df = pd.read_csv("HKdata.data", header=None, delimiter=',')
xtrain_files1, ytrain_files1, xtest_files1, ytest_files1, X_test1, Y_test1 = splitdata(dataMY, 0.1)
xtrain_files2, ytrain_files2, xtest_files2, ytest_files2, X_test2, Y_test2 = splitdata(dataBO, 0.1)
xtrain_files3, ytrain_files3, xtest_files3, ytest_files3, X_test3, Y_test3 = splitdata(dataHK, 0.1)


metricarrs=[[1,"poly"],[1,"rbf"],[1,"sigmoid"],
[2,"poly"],[2,"rbf"],[2,"sigmoid"],
[3,"poly"],[3,"rbf"],[3,"sigmoid"],
[4,"poly"],[4,"rbf"],[4,"sigmoid"],
[5,"poly"],[5,"rbf"],[5,"sigmoid"]]
metricarrsplot=["1 and poly","1 and rbf","1 and sigmoid",
"2 and poly","2 and rbf","2 and sigmoid",
"3 and poly","3 and rbf","3 and sigmoid",
"4 and poly","4 and rbf","4 and sigmoid",
"5 and poly","5 and rbf","5 and sigmoid"]
myaccy=[]
boaccy=[]
hkaccy=[]
for metricarr in metricarrs:
    neigh1 = sklearn.svm.SVC(C=metricarr[0],kernel=metricarr[1])
    neigh2 = sklearn.svm.SVC(C=metricarr[0],kernel=metricarr[1])
    neigh3 = sklearn.svm.SVC(C=metricarr[0],kernel=metricarr[1])
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