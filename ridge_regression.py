# Ridge Regression
# Importing required libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# reading the train data
X_train = np.loadtxt("data/X_train.csv",delimiter=",")
y_train = np.loadtxt("data/y_train.csv",delimiter=",")

# standardizing X_train
x_scaler1 = StandardScaler().fit(X_train[:,0:6])
X_train = x_scaler1.transform(X_train[:,0:6])

# fitting degree 2 scaler
b = np.power(X_train[:, 0:6], 2)
x_scaler2 = StandardScaler().fit(b[:,0:6])

# fitting degree 3 scaler
c = np.power(X_train[:, 0:6], 3)
x_scaler3 = StandardScaler().fit(c[:,0:6])

def RidgeRegression(X, y):
    # creating empty lists to store results
    wRR = []
    df = []
    for i in range(0, 5001, 1):
        # initializing lambda parameter
        lmbd = i
        # X^T
        xtrans = np.transpose(X)
        # X^T * X
        xTx = np.dot(xtrans, X)
        # lambda * I
        lmbd_I = np.identity(xTx.shape[0]) * lmbd
        # (lambda * I + X^T * X) ^ -1
        inverse = np.linalg.inv(lmbd_I + xTx)
        # X^T * y
        xTy = np.dot(xtrans, y)
        # (lambda * I + X^T * X) ^ -1 * X^T * y
        wRR_i = np.dot(inverse, xTy)
        # finding svd
        U, S, V = np.linalg.svd(X)
        # df(lambda)
        df_i = np.sum(np.square(S) / (np.square(S) + lmbd))
        # appending to lists
        wRR.append(wRR_i)
        df.append(df_i)
    return np.asarray(wRR), np.asarray(df)


# calling the ridge regression function
wRR, df = RidgeRegression(X_train, y_train)

plt.figure(figsize=(12,8))### Part 1. a)
# storing colors and labels for the graph
colors = ['#96ceb4','#ffcc5c','#ff6f69','#ced07d','#0e9aa7','#aa7eb2','#698d70']
labels = ["Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Year Made", "Intercept"]
# plotting line for each feature
for i in range(0, wRR[0].shape[0]):
    plt.plot(df, wRR[:,i], color = colors[i])
    plt.scatter(df, wRR[:,i], color = colors[i], s=12 , label=labels[i])
plt.xlabel(r"df($\lambda$)")
plt.legend(loc='lower left')
plt.show()

X_test = np.loadtxt('hw1-data/X_test.csv', delimiter=',')
y_test = np.loadtxt('hw1-data/y_test.csv', delimiter=',')

# standardizing X_test
X_test = x_scaler1.transform(X_test[:,0:6])

def calcRMSE(X_test, y_test, wRR, max_lmbd, poly):
    # creating list to store RMSE values
    RMSE = []
    for lmbd in range(0, max_lmbd + 1):
        # getting corresponding WRR vector
        wRR_i = wRR[lmbd]
        # predicting for X_test
        y_pred = np.dot(X_test, wRR_i)
        # calculating RMSE and appending to list
        RMSE_i = np.sqrt(np.sum(np.square(y_test - y_pred))/len(y_test))
        RMSE.append(RMSE_i)
    # storing colors and labels for graph
    colors = ['#96ceb4','#ffcc5c','#ff6f69','#ced07d']
    legend = ["Polynomial Order, p = 1", "Polynomial Order, p = 2", "Polynomial Order, p = 3"]
    # plotting required curve
    plt.plot(range(len(RMSE)), RMSE, color = colors[poly])
    plt.scatter(range(len(RMSE)), RMSE, color = colors[poly] , s = 12, label= legend[poly-1])
    plt.xlabel(r"$\lambda$")
    plt.ylabel("RMSE")
    plt.legend(loc='upper left')
    plt.title(r"RMSE vs $\lambda$ values for the test set, $\lambda$ = 0....%d" %(max_lmbd))


plt.figure(figsize=(12,8))
# calling rmse for polynomial order = 1
calcRMSE(X_test, y_test, wRR, 50, poly=1)
plt.show()


def PolyOrderInput(input_data, p):
    if p == 1:
        return input_data
    elif p == 2:
        a = input_data
        b = np.power(input_data[:, 0:6], 2)
        # standardizing degree 2 columns
        b = x_scaler2.transform(b[:,0:6])
        new_data = np.hstack((a, b))
        return new_data
    elif p == 3:
        a = input_data
        b = np.power(input_data[:, 0:6], 2)
        # standardizing degree 2 columns
        b = x_scaler2.transform(b[:,0:6])
        c = np.power(input_data[:, 0:6], 3)
        # standardizing degree 3 columns
        c = x_scaler3.transform(c[:,0:6])
        new_data = np.hstack((a, b, c))
        return new_data


plt.figure(figsize=(12,8))
for i in [1, 2, 3]:
    X_train_appended = PolyOrderInput(X_train, p = i)
    X_test_appended = PolyOrderInput(X_test, p = i)
    wRR, df = RidgeRegression(X_train_appended, y_train)
    calcRMSE(X_test_appended, y_test, wRR, 100, poly=i)
plt.show()
