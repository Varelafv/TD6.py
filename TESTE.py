import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import math

def studentTheorique(v,n):
    moy=np.mean(v)
    sigma=np.sqrt(np.var(v))
    x=np.linspace(-10,10,n)
    y=sc.norm.pdf(x,moy,sigma)
    return x,y

def myTtest(v1,v2,alpha):
    n1=len(v1)
    n2=len(v2)
    moy1=np.mean(v1)
    moy2=np.mean(v2)
    var1=np.sum(pow(v1,2)-pow(moy1,2))/n1
    var2=np.sum(pow(v2,2)-pow(moy2,2))/n2
    T=(moy1-moy2)/np.sqrt(var1/n1 + var2/n2)
    beta=sc.t.cdf(T,n1-1)
    if (1-alpha/2)>beta:
        #print("accepte")
        decision=1
    else:
        #print("rejette")
        decision=0
    return decision,T,beta

#question c)
plt.figure(1)
alpha=0.05
T1=np.zeros(500,dtype='float')
for i in range(500):
    X1=np.random.normal(0,2,500)
    X2=np.random.normal(0,2,500)
    T1[i]=myTtest(X1, X2, alpha)[1]
x1,T2=myEpdf(T1)
plt.plot(x1,T2)
xTheo1,Theo1=studentTheorique(T1,500)
plt.plot(xTheo1,Theo1)