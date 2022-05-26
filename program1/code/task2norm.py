import  time
from turtle import clear
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn import preprocessing



data = pd.read_excel("Concrete_Data.xls")


data.insert(0, 'x0', 1, allow_duplicates=False)

datax=preprocessing.scale(data.iloc[:,:-1])
# print(datax)

datax[:1031,0]=1
xtrain=datax[:900,:]
ytrain=data.iloc[0:900,-1]
tarinvariance= np.var(ytrain)
xtest=datax[900:1031,:]
ytest=data.iloc[900:1031,-1]
testvariance= np.var(ytest)

# xtrainnor=data_normal.iloc[:900,:-1]
# xtestnor=data_normal.iloc[900:1031,:-1]

ytest.index-=900
initial_m=[0]*9
alpha=0.1
num_iter=10000
# print(xtrain[2:3]["Cement (component 1)(kg in a m^3 mixture)"],ytrain[1])
# print(xtrain.iloc[1:3,0:2],ytrain[1])
# print(xtest,ytest)


def compute_cost(m,xdata,ydata):
    total_cost=0
    M = len(xdata)
    for i in range(M):
        y=ydata[i]
        curcost=0
        for j in range(len(m)):
            # print(xtrain.iloc[i:i+1,j:j+1])
            # print(m[j])
            curcost+=xdata[i][j]*m[j]
        total_cost += (y-curcost)**2
        # print(total_cost)
    return total_cost/M



# compute_cost(m,xtrain,ytrain)
def step_grad_desc(current_m,alpha,xdata,ydata):
    updated_m=[0]*9
    for i in range(len(current_m)):
        sum_grad_m=0
        M=float(len(xdata))
        for j in range(len(xdata)):
            x= xdata[j]
            y= ydata[j]
            curxsum=0
            for m in range(len(current_m)):
                curxsum+= x[m]*current_m[m]
            sum_grad_m += (curxsum-y) * x[i]
        grad_m=2/M * sum_grad_m      
        updated_m[i] = current_m[i]- alpha * grad_m
    return updated_m

def grad_desc(xdata,ydata,initial_m,alpha,num_iter):
    m = initial_m
    cost_list=[]
    for i in range(num_iter):
        print("---------------------------------------------")
        print("iteration:",i)
        cost_list.append(compute_cost(m,xdata,ydata))
        m= step_grad_desc(m,alpha,xdata,ydata)
    return [m,cost_list]

m,cost_list= grad_desc(xtrain,ytrain,initial_m,alpha,num_iter)
print ("m is :",m)
# print(xtrain,ytrain)
cost = compute_cost(m,xtest,ytest)

print(cost_list[-2])
print(cost_list[-1])
# print("cost_list:",cost_list)
print(1-cost_list[-1]/tarinvariance)
print("cost is:",cost)
print(1-cost/testvariance)
