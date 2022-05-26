import imp
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from dataprocess import splitdata
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

dataMY = df = pd.read_csv("MYdata.data", header=None, delimiter=',')
dataBO = df = pd.read_csv("BOdata.data", header=None, delimiter=',')
dataHK = df = pd.read_csv("HKdata.data", header=None, delimiter=',')
xtrain_files1, ytrain_files1, xtest_files1, ytest_files1, X_test1, Y_test1 = splitdata(dataMY, 0.1)
xtrain_files2, ytrain_files2, xtest_files2, ytest_files2, X_test2, Y_test2 = splitdata(dataBO, 0.1)
xtrain_files3, ytrain_files3, xtest_files3, ytest_files3, X_test3, Y_test3 = splitdata(dataHK, 0.1)

metricarrs=[[20,0.5],[20,1.0],[20,1.5],[20,2.0],[20,2.5],
[50,0.5],[50,1.0],[50,1.5],[50,2.0],[50,2.5],
[100,0.5],[100,1.0],[100,1.5],[100,2.0],[100,2.5],
[150,0.5],[150,1.0],[150,1.5],[150,2.0],[150,2.5],
[200,0.5],[200,1.0],[200,1.5],[200,2.0],[200,2.5]]
metricarrsplot=["20 and 0.5","20 and 1.0","20 and 1.5","20 and 2.0","20 and 2.5",
"50 and 0.5","50 and 1.0","50 and 1.5","50 and 2.0","50 and 2.5",
"100 and 0.5","100 and 1.0","100 and 1.5","100 and 2.0","100 and 2.5",
"150 and 0.5","150 and 1.0","150 and 1.5","150 and 2.0","150 and 2.5",
"200 and 0.5","200 and 1.0","200 and 1.5","200 and 2.0","200 and 2.5"]
myaccy=[]
boaccy=[]
hkaccy=[]
for metricarr in metricarrs:
    neigh1 = AdaBoostClassifier(n_estimators=metricarr[0], learning_rate=metricarr[1])
    neigh2 = AdaBoostClassifier(n_estimators=metricarr[0], learning_rate=metricarr[1])
    neigh3 = AdaBoostClassifier(n_estimators=metricarr[0], learning_rate=metricarr[1])
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