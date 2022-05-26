import  time
from turtle import clear
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn import preprocessing


data = pd.read_excel("Concrete_Data.xls").to_numpy()


# xtrain=data.iloc[:,:-1]

x = data[:, 7].reshape(-1 , 1)
std = preprocessing.scale(x)
plt.hist(std, label=f"feature_{7}")
plt.legend()
plt.show()
