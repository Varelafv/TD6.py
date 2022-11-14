import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import math
from math import sqrt
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

Stats = np.zeros ((1000, 2), float)
def Corr (a,b) :

    for i in range(1000):
        X= np.random.randint(0, 2, 100) #création du tableau X
        v = np.random.randint(0, 2, 100)
        Y= a * X + b + v
        Data =  ([X, Y])
        c,p,t = myRho(Data,100)
        Stats[i,0]=p
        Stats[i,1]=t
    return Stats
Stats1 = Corr (0, 0)
counts, bins = np.histogram(Stats1[:,1], 8)
plt.plot(1)
plt.title('Histograma 1000 valeurs de ρ(X, Y ) ')
plt.text(4,250, r'$a=1,b=0$')
plt.grid(True)
plt.hist(bins[:-1], bins, weights = counts,color = "red", ec="black")


Stats1 = Corr (-1, 0)
counts, bins = np.histogram(Stats1[:,1], 8)
plt.plot(1)
plt.title('Histograma 1000 valeurs de ρ(X, Y ) ')
plt.text(-5.0,350, r'$a=-1,b=0$')
plt.grid(True)
plt.hist(bins[:-1], bins, weights = counts,color = "yellow", ec="blue")



t1 = np.linspace (-15, 15, 1000)
lei = sc.t.pdf (t1, 998)*1000
plt.plot (3)
plt.plot (t1, lei, 'b')
t2 = [25, 50, 100, 250, 500, 1000]
for i in range (6):
    st_law_cumul = sc.t.cdf (t1, t2[i]-2)*400
    plt.plot (t1, st_law_cumul, 'r')
""" 
Stats2 = Corr (-1, 0)
counts, bins = np.histogram(Stats2[:,1], 8)
plt.plot(2)
plt.hist(bins[:-1], bins, weights = counts)

"""
plt.show()
