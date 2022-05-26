import  time
from turtle import clear
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_excel("Concrete_Data.xls")

datax=preprocessing.scale(data.iloc[:,7])

xtrain=data.iloc[0:900,7]
ytrain=data.iloc[0:900,-1]
tarinvariance= np.var(ytrain)
xtest=data.iloc[900:1031,7]
ytest=data.iloc[900:1031,-1]
testvariance= np.var(ytest)


alpha = 0.00000001
initial_m = 0
initial_b = 36
num_iter =10000


def compute_cost(m,b,xdata,ydata):
    total_cost=0
    M = len(xdata)
    for i in range(M):
        x=xdata[i]
        y=ydata[i]
        total_cost += (y-m*x-b)**2
    return total_cost/M

def compute_costtest(m,b,xdata,ydata):
    total_cost=0
    M = len(xdata)
    for i in range(900,900+M):
        x=xdata[i]
        y=ydata[i]
        total_cost += (y-m*x-b)**2
    return total_cost/M


def step_grad_desc(current_m,current_b,alpha,xdata,ydata):
    sum_grad_m=0
    sum_grad_b=0
    M=len(xdata)
    for i in range(M):
        x= xdata[i]
        y= ydata[i]
        sum_grad_m += (current_m * x +current_b -y) *x
        sum_grad_b +=  current_m * x +current_b -y
    grad_m=2/M * sum_grad_m
    grad_b=2/M * sum_grad_b
    
    updated_m = current_m- alpha * grad_m
    updated_b = current_b -alpha * grad_b
    return updated_m,updated_b


def grad_desc(xdata,ydata,initial_m,initial_b,alpha,num_iter):
    m = initial_m
    b = initial_b
    cost_list=[]
    for i in range(num_iter):
        print("--------------------------------")
        print("iteration:",i)
        cost_list.append(compute_cost(m,b,xdata,ydata))
        m,b= step_grad_desc(m,b,alpha,xdata,ydata)
    return [m,b,cost_list]


m,b,cost_list= grad_desc(xtrain,ytrain,initial_m,initial_b,alpha,num_iter)
print ("m is :",m)
print ("b is :",b)
# print(xtrain,ytrain)
cost = compute_costtest(m,b,xtest,ytest)

# print("cost_list:",cost_list)
print(cost_list[-2])
print(cost_list[-1])
print(1-cost_list[-1]/tarinvariance)
print("cost is:",cost)
print(1-cost/testvariance)

plt.scatter(xtrain,ytrain,s=0.5)

pred_y= m*xtrain+b

plt.plot(xtrain,pred_y,c='r')

plt.show()