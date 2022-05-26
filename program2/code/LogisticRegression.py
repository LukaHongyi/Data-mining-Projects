import imp
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from dataprocess import splitdata
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

dataMY = df = pd.read_csv("MYdata.data", header=None, delimiter=',')
dataBO = df = pd.read_csv("BOdata.data", header=None, delimiter=',')
dataHK = df = pd.read_csv("HKdata.data", header=None, delimiter=',')
xtrain_files1, ytrain_files1, xtest_files1, ytest_files1, X_test1, Y_test1 = splitdata(dataMY, 0.1)
xtrain_files2, ytrain_files2, xtest_files2, ytest_files2, X_test2, Y_test2 = splitdata(dataBO, 0.1)
xtrain_files3, ytrain_files3, xtest_files3, ytest_files3, X_test3, Y_test3 = splitdata(dataHK, 0.1)

metricarrs=[[100,"liblinear"],[100,"lbfgs"],[100,"newton-cg"],[100,"sag"],
[200,"liblinear"],[200,"lbfgs"],[200,"newton-cg"],[200,"sag"],
[300,"liblinear"],[300,"lbfgs"],[300,"newton-cg"],[300,"sag"],
[400,"liblinear"],[400,"lbfgs"],[400,"newton-cg"],[400,"sag"],
[500,"liblinear"],[500,"lbfgs"],[500,"newton-cg"],[500,"sag"]]
metricarrsplot=["100 and liblinear","100 and lbfgs","100 and newton-cg","100 and sag",
"200 and liblinear","200 and lbfgs","200 and newton-cg","200 and sag",
"300 and liblinear","300 and lbfgs","300 and newton-cg","300 and sag",
"400 and liblinear","400 and lbfgs","400 and newton-cg","400 and sag",
"500 and liblinear","500 and lbfgs","500 and newton-cg","500 and sag"]
myaccy=[]
boaccy=[]
hkaccy=[]
for metricarr in metricarrs:
    neigh1 = LogisticRegression(max_iter=metricarr[0], solver=metricarr[1])
    neigh2 = LogisticRegression(max_iter=metricarr[0], solver=metricarr[1])
    neigh3 = LogisticRegression(max_iter=metricarr[0], solver=metricarr[1])
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