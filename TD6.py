import numpy as np
import csv
import matplotlib.pyplot as plt
from  math import sqrt
import scipy .stats as sc


def myRho(X, Y):
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

    # Ecart-type :
    Ecart_x = np.sqrt(Var_x)
    Ecart_y = np.sqrt(Var_y)

    # Correlation :
    Rho_xy = Cov_xy / (Ecart_x * Ecart_y)

    # Statistique T :
   # T = Rho_xy / np.sqrt((1 - Rho_xy ** 2) / (len(X) - 2))

    return (Cov_xy,Var_x )

def studentTheorique(v,n):
    moy=np.mean(v)
    sigma=np.sqrt(np.var(v))

    return x,y
def myRegLin( X,Y ):
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
    B1= Cov_xy /  Var_x
    B0= E_y - B1*np.mean(X)
    #calcul des residus
    E= Y - (B0 + B1 * X)
    V= np.mean(E**2)  -  np.mean(E)**2
    R= Cov_xy**2/(Var_x*Var_y)
    Yr=B1*X+B0
    p=sqrt(R)
    print(p)
    print(np.mean(X),np.mean(Yr))
    plt.plot(X,Y,'o')
    plt.plot(x, Yr)
    plt.plot(np.mean(X),np.mean(Yr) ,'ro')

   # count, bins = np.histogram(E, 10)
    #plt.hist(bins[:-1], bins, weights=count, color="red", edgecolor="black", density=True)"""
    return(B0,B1,R,V)
#Y=np.random.uniform(0,10,1000),
x=np.random.randint(0,11,1000)
n=np.random.normal(0,3,1000)
y=2*x + 3 + n
B0,B1,R,V=myRegLin(x,y)

plt.show()