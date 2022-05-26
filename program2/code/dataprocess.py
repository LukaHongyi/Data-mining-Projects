import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn import manifold
from sklearn.decomposition import FactorAnalysis, FastICA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import imp
 


def text_save(filename, data, P1, P2):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    data = np.array(data)
    for i in range(len(data)):
        if (data[i][0][0] == P1) or (data[i][0][0] == P2):
            s = data[i][0]+'\n'
            file.write(s)
        else: continue
    file.close()
    print("file save")

# data = initData("BOdata.data")
# text_save("HKdata.data", data, "K", "H")
# text_save("MYdata.data", data, "M", "Y")
# text_save("BOdata.data", data, "B", "O")

def splitdata(data, percentage):
    X = data.iloc[:,1:]
    Y = data.iloc[:,0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=percentage, random_state=0)
    xtrain_files, xtest_files = [], []
    ytrain_files, ytest_files = [], []
    # print(type(X_train))
    # print(len(X_train))
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X_train):
        # print("%s %s" % (train_index, test_index))
        xtrain_files.append(np.array(X_train)[train_index].tolist())
        xtest_files.append(np.array(X_train)[test_index].tolist())

    for train_index, test_index in kf.split(Y_train):
        # print("%s %s" % (train_index, test_index))
        ytrain_files.append(np.array(Y_train)[train_index].tolist())
        ytest_files.append(np.array(Y_train)[test_index].tolist())
    
    return xtrain_files, ytrain_files, xtest_files, ytest_files, X_test, Y_test
    
def lowvariance(data):
    X = data.iloc[:,1:]
    Y = data.iloc[:,0]
    var = X.var()
    index = np.argpartition(var, -4)[-4:]
    datamerge = pd.concat([Y,X[index]], axis=1)
    print(datamerge)
    return datamerge

def highcorr(data):
    index_column=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    X = data.iloc[:,1:]
    Y = data.iloc[:,0]    
    corr = X.corr()
    corr=np.array(corr)
    threadhold = 0.5
    while len(index_column)>4:
        threadhold-=0.01
        for i in index_column:
            for j in index_column:
                if corr[i][j] > threadhold and i != j:  
                    print(i,j,corr[i][j])
                    index_column.remove(j)
    new_columns = index_column
    # print(new_columns)
    # print(var)
    # index = np.argpartition(var, -4)[-4:]
    datamerge = pd.concat([Y,X[new_columns]], axis=1)
    print(datamerge)
    return datamerge

def randomforests(data):
    X = data.iloc[:,1:]
    Y = data.iloc[:,0]    
    for i in range(len(Y)):
        Y[i] = ord(Y[i])
    model = RandomForestRegressor(random_state=1, max_depth=10)
    df=pd.get_dummies(X)
    model.fit(df,Y)
    features = df.columns
    importances = model.feature_importances_
    index = np.argpartition(importances, -4)[-4:]
    datamerge = pd.concat([Y,X[index]], axis=1)
    print(datamerge)
    return datamerge

# def Backward(data):
#     X = data.iloc[:,1:]
#     Y = data.iloc[:,0]

#     for i in range(len(Y)):
#         Y[i] = ord(Y[i])
#     lreg = LinearRegression()
#     rfe = RFE(estimator = lreg, n_features_to_select=4)
#     rfe = rfe.fit_transform(X, Y)
#     rfe = pd.DataFrame(rfe)
#     datamerge = pd.concat([Y,rfe], axis=1)
#     print(datamerge)

def factorAnalysis(data):
    X = data.iloc[:,1:]
    Y = data.iloc[:,0]  
    fa = FactorAnalysis(n_components=4)  
    fa.fit(X)
    tran_x = fa.transform(X)
    factor_columns = []
    for index in range(4):
        tmp = "factor" + str(index + 1)
        factor_columns.append(tmp)
    tran_df = pd.DataFrame(tran_x, columns=factor_columns)
    datamerge = pd.concat([Y,tran_df], axis=1)
    print(datamerge)
    return datamerge

def pca(data):
    X = data.iloc[:,1:]
    Y = data.iloc[:,0]
    pca = decomposition.PCA(n_components=4)  
    pca.fit(X)
    print(pca.explained_variance_ratio_)  
    print(pca.explained_variance_)  
    print(pca.n_features_)
    print(pca.n_features_in_)
    newdf = pca.fit_transform(X)
    factor_columns = []
    for index in range(4):
        tmp = "factor" + str(index + 1)
        factor_columns.append(tmp)
    tran_df = pd.DataFrame(newdf, columns=factor_columns)
    datamerge = pd.concat([Y,tran_df], axis=1)
    print(datamerge)
    return datamerge


def ica(data):
    X = data.iloc[:,1:]
    Y = data.iloc[:,0]
    ICA = FastICA(n_components=4, random_state=16)
    newdf = ICA.fit_transform(X)
    factor_columns = []
    for index in range(4):
        tmp = "factor" + str(index + 1)
        factor_columns.append(tmp)
    tran_df = pd.DataFrame(newdf, columns=factor_columns)
    datamerge = pd.concat([Y,tran_df], axis=1)
    print(datamerge)
    return datamerge


def svd(data):
    X = data.iloc[:,1:]
    Y = data.iloc[:,0]
    newdf = TruncatedSVD(n_components=4, random_state=16).fit_transform(X)
    factor_columns = []
    for index in range(4):
        tmp = "factor" + str(index + 1)
        factor_columns.append(tmp)
    tran_df = pd.DataFrame(newdf, columns=factor_columns)
    datamerge = pd.concat([Y,tran_df], axis=1)
    print(datamerge)
    return datamerge

def IOSMAP(data):
    X = data.iloc[:,1:]
    Y = data.iloc[:,0]
    newdf = manifold.Isomap(n_neighbors=5, n_components=4, n_jobs=-1).fit_transform(X)
    factor_columns = []
    for index in range(4):
        tmp = "factor" + str(index + 1)
        factor_columns.append(tmp)
    tran_df = pd.DataFrame(newdf, columns=factor_columns)
    datamerge = pd.concat([Y,tran_df], axis=1)
    print(datamerge)
    return datamerge


# data = df = pd.read_csv("BOdata.data", header=None, delimiter=',')
# tsne(data)




# # print(data)
# neigh = KNeighborsClassifier(n_neighbors=2)
# xtrain_files, ytrain_files, xtest_files, ytest_files, X_test, Y_test = splitdata(data, 0.1)
# acc=0
# for i in range(5):
#     neigh.fit(xtrain_files[i], ytrain_files[i])
#     test_predictions = neigh.predict(xtest_files[i])
#     acc += accuracy_score(ytest_files[i], test_predictions)
#     print(accuracy_score(ytest_files[i], test_predictions))
# acc = acc/5
# print(acc)