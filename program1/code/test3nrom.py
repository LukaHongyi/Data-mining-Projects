import  time
from turtle import clear
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from itertools import combinations
from sklearn import preprocessing

data = pd.read_excel("Concrete_Data.xls")


datax=preprocessing.scale(data.iloc[:,:-1])
xtrain=datax[:900,:]
ytrain=data.iloc[0:900,-1]
tarinvariance= np.var(ytrain)
xtest=datax[900:1031,:]
ytest=data.iloc[900:1031,-1]
testvariance= np.var(ytest)
ytest.index-=900


initial_m=[0]*45
alpha=0.01
num_iter=10000

def compute_cost(m,xdata,ydata):
    total_cost=0
    M = len(xdata)
    for i in range(M):
        y=ydata[i]
        curcost=compute_curcost(m,xdata[i])
        total_cost += (y-curcost)**2
    return total_cost/float(M)


def compute_curcost(m,xcurdata):
    mcount=0
    curcost=0
    for j in range(8):
        for n in range(j,8):
            curcost+=xcurdata[j]*xcurdata[n]*m[mcount]
            mcount+=1
    for h in range(8):
        curcost+=xcurdata[h]*m[mcount]
        mcount+=1
    curcost+=m[-1]
    return curcost

def mpos(pos):
    if pos<=7:
        return 0,pos
    elif pos<=14:
        return 1,pos-7
    elif pos<=20:
        return 2,pos-13
    elif pos<=25:
        return 3,pos-18
    elif pos<=29:
        return 4,pos-22
    elif pos<=32:
        return 5,pos-25
    elif pos<=34:
        return 6,pos-27
    else:
        return 7,7
    

def step_grad_desc(current_m,alpha,xdata,ydata):
    updated_m=[0]*45
    for i in range(len(current_m)):
        sum_grad_m=0
        M=float(len(xdata))
        for j in range(len(xdata)):
            x= xdata[j]
            y= ydata[j]
            curxsum=compute_curcost(current_m,x)
            if i==44:
                sum_grad_m += (curxsum-y)
            elif i>35:
                sum_grad_m += (curxsum-y) * x[i-36]
            else:
                pos1,pos2=mpos(i)
                sum_grad_m += (curxsum-y) * x[pos1]*x[pos2]
        grad_m=2/M * sum_grad_m      
        updated_m[i] = current_m[i]- alpha * grad_m
    return updated_m


def grad_desc(xdata,ydata,initial_m,alpha,num_iter):
    m = initial_m
    cost_list=[]
    for i in range(num_iter):
        print("--------------------------------")
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
print(1-cost_list[-1]/tarinvariance)
print("cost is:",cost)
print(1-cost/testvariance)
# # plt.plot(cost_list)
# # print(compute_cost(m,xtrain,ytrain))
