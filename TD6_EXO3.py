import numpy as np

import matplotlib.pyplot as plt
from  math import *
import scipy .stats as sc


def myRegLin(X,Y ):
    Y1=Y
    Y=np.log(np.abs(Y))

    # Espérence :
    E_x = np.mean(X)
    E_xx = np.mean(X ** 2)
    E_y = np.mean(Y)
    E_yy = np.mean(Y ** 2)
    E_xy = np.mean(X * Y)

    # Covariance :
    Cov_xy = E_xy - E_x * E_y

    # Variance :
    Var_x = E_xx - (E_x ** 2)
    Var_y = E_yy - (E_y ** 2)
    B1 = Cov_xy / Var_x
    B0 = E_y - B1 * np.mean(X)
    # calcul des residus
    E = Y - (B0 + B1 * X)
    V = np.mean(E ** 2) - np.mean(E) ** 2
    R = Cov_xy ** 2 / (Var_x * Var_y)
    Yr = np.exp(B0) * np.exp(B1*X)
    p = sqrt(R)
    plt.plot(X, Y1, 'o')
    plt.plot(X, Yr,'ro')
    #plt.plot(np.mean(X), np.mean(Yr), 'ro')
    # plt.figure(1)
    count, bins = np.histogram(E, 10)
    # plt.hist(bins[:-1], bins, weights=count, color="red", edgecolor="black", density=True)
    return (B0, B1, R, V)
#Y=np.random.uniform(0,10,1000),
#x=np.random.randint(0,11,1000)
n=np.random.normal(0,3,1000)
#y=2*x + 3 + n
#B0,B1,R,V=myRegLin(x,y)
#alpha=0.05
#n=len(x)
#y1=interval_pre(B0,B1,alpha,n,x,y)
x= np.load('X-TD6-EX3.npy', mmap_mode='r')
y= np.load('Y-TD6-EX3.npy', mmap_mode='r')
print(y)
B0,B1,R,V=myRegLin(x,y)
plt.show()