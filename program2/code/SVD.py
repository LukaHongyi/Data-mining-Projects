import imp
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from dataprocess import splitdata,IOSMAP
from sklearn.ensemble import RandomForestClassifier
import math
import matplotlib.pyplot as plt

dataMY = df = pd.read_csv("MYdata.data", header=None, delimiter=',')
dataBO = df = pd.read_csv("BOdata.data", header=None, delimiter=',')
dataHK = df = pd.read_csv("HKdata.data", header=None, delimiter=',')
dataMY = IOSMAP(dataMY)
dataBO = IOSMAP(dataBO)
dataHK = IOSMAP(dataHK)
xtrain_files1, ytrain_files1, xtest_files1, ytest_files1, X_test1, Y_test1 = splitdata(dataMY, 0.1)
xtrain_files2, ytrain_files2, xtest_files2, ytest_files2, X_test2, Y_test2 = splitdata(dataBO, 0.1)
xtrain_files3, ytrain_files3, xtest_files3, ytest_files3, X_test3, Y_test3 = splitdata(dataHK, 0.1)

metricarrs=[[3,1],[3,2],[3,3],[3,4],[3,5],
[4,1],[4,2],[4,3],[4,4],[4,5],
[5,1],[5,2],[5,3],[5,4],[5,5],
[6,1],[6,2],[6,3],[6,4],[6,5],
[7,1],[7,2],[7,3],[7,4],[7,5]]
metricarrsplot=["3 and 1","3 and 2","3 and 3","3 and 4","3 and 5",
"4 and 1","4 and 2","4 and 3","4 and 4","4 and 5",
"5 and 1","5 and 2","5 and 3","5 and 4","5 and 5",
"6 and 1","6 and 2","6 and 3","6 and 4","6 and 5",
"7 and 1","7 and 2","7 and 3","7 and 4","7 and 5"]
myaccy=[]
boaccy=[]
hkaccy=[]
for metricarr in metricarrs:
    neigh1 = RandomForestClassifier(min_samples_split=metricarr[0],min_samples_leaf=metricarr[1])
    neigh2 = RandomForestClassifier(min_samples_split=metricarr[0],min_samples_leaf=metricarr[1])
    neigh3 = RandomForestClassifier(min_samples_split=metricarr[0],min_samples_leaf=metricarr[1])
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


