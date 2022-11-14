import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import scipy.stats as sc
def myRho(T,n):
    COV=0;
    pq =0 ;
    SY=0;
    SX=0;
    EX=sum(T[0][0:])/float(len(T[0][0:])) #La experence de x
    EY=sum(T[1][0:])/float(len(T[1][0:])) #La experence de y
    for i in range(n):
        COV = COV + (T[0][i] - EX) * (T[1][i] - EY)
        SX = SX + (T[0][i] - EX) ** 2
        SY = SY + (T[1][i] - EY) ** 2
    pq = COV /sqrt(SX * SY) #La correlation
    COV = COV / n #La covolution de x y
    t=pq/sqrt((1-pq**2)/(n-2)) #La statistique sur la correlation de x y
    return (COV,pq,t)
def mysort(X) :
    for b in range(0,2):
        GR = 0
        T = []
        T=X[b][0:]
        Tu = np.unique(T)
        if b==0 :
            GRAX = np.zeros(np.shape(T), dtype='float')
        if b==1 :
              GRAX1 = np.zeros(np.shape(T), dtype='float')
        for i in Tu:
            iNd = np.where(T == i)[0]
            nx = len(iNd)
            if b==0:
                GRAX[iNd] = np.mean(np.arange(GR, GR + nx))
                GR = GR + nx
            if b==1:
                GRAX1[iNd] = np.mean(np.arange(GR, GR + nx))
                GR = GR + nx
    return(GRAX,GRAX1)


X = np.random.uniform (0, 12, 500)
Y=np.exp(X)+np.random.normal(0,1,500)

M = np.zeros((500, 2));
A = np.random.randint(0, 15, 10)

B=np.random.randint(0, 15, 10)

C=list([X,Y])
D,R=mysort(C)
print(C)
print(D,R)
plt.figure(2)
plt.scatter(D,R, marker='+')
C=list([D,R])
a=myRho(C,500)
print(a)
plt.show()
