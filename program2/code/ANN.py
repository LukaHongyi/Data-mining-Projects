import imp
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from dataprocess import splitdata
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

dataMY = df = pd.read_csv("MYdata.data", header=None, delimiter=',')
dataBO = df = pd.read_csv("BOdata.data", header=None, delimiter=',')
dataHK = df = pd.read_csv("HKdata.data", header=None, delimiter=',')
xtrain_files1, ytrain_files1, xtest_files1, ytest_files1, X_test1, Y_test1 = splitdata(dataMY, 0.1)
xtrain_files2, ytrain_files2, xtest_files2, ytest_files2, X_test2, Y_test2 = splitdata(dataBO, 0.1)
xtrain_files3, ytrain_files3, xtest_files3, ytest_files3, X_test3, Y_test3 = splitdata(dataHK, 0.1)

metricarrs=[[100,"relu"],[100,"logistic"],[100,"tanh"],[100,"identity"],
[200,"relu"],[200,"logistic"],[200,"tanh"],[200,"identity"],
[300,"relu"],[300,"logistic"],[300,"tanh"],[300,"identity"],
[400,"relu"],[400,"logistic"],[400,"tanh"],[400,"identity"],
[500,"relu"],[500,"logistic"],[500,"tanh"],[500,"identity"]]
metricarrsplot=["100 and relu","100 and logistic","100 and tanh","100 and identity",
"200 and relu","200 and logistic","200 and tanh","200 and identity",
"300 and relu","300 and logistic","300 and tanh","300 and identity",
"400 and relu","400 and logistic","400 and tanh","400 and identity",
"500 and relu","500 and logistic","500 and tanh","500 and identity"]
myaccy=[]
boaccy=[]
hkaccy=[]
for metricarr in metricarrs:
    neigh1 = MLPClassifier(hidden_layer_sizes=(400,100),alpha=0.01,max_iter=metricarr[0],activation=metricarr[1])
    neigh2 = MLPClassifier(hidden_layer_sizes=(400,100),alpha=0.01,max_iter=metricarr[0],activation=metricarr[1])
    neigh3 = MLPClassifier(hidden_layer_sizes=(400,100),alpha=0.01,max_iter=metricarr[0],activation=metricarr[1])
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