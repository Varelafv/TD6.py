import numpy as np
from myRho import *
import matplotlib.pyplot as plt
from math import *

"""EXERCICE 1 """
def myProba(x):
    Tu = np.unique(x)
    q = len(x);  # le nombre de d'apparition de valeur mazeer
    T1=np.zeros((2,len(Tu)))
    ind=0
    T1[0][0:]=Tu
    for i in Tu:
        s=0 ;
        for b in x:
            if b==i :
                s=s+1
        T1[1][ind]=float(s)/float(q)
        ind = ind + 1
    return (T1)
def myProbajoint(T1,T2):
    T1Uni=np.unique(T1)
    T2Uni=np.unique(T2)
    P=np.zeros((len(T1Uni),len(T2Uni)),dtype = float)
    for i in range (np.size(T1)):
        P[np.where(T1Uni==T1[i]),np.where(T2Uni == T2[i])]= P[np.where(T1Uni==T1[i]), np.where(T2Uni == T2[i])]+1
        P= P/len(T1)
        plt.imshow(P)
        plt.colorbar()
        return (P)
def  myEntropy (T):
    T_P= np.zeros(len(T))
    T_P=myProba(T)
    Entropy = 0
    for i in range(np.shape(T_P)[0]):
        Entropy = Entropy -T_P[0][i]*np.log2(T_P[1][i])
    return Entropy
def myEntropyJointe(T1,T2) :
    ProbJ = np.zeros(len(T1))
    ProbJ = myProbajoint(T1,T2)
    Entropy = 0
    for i in range(np.shape(ProbJ)[0]):
        for j in range(np.shape(ProbJ)[1]):
            if ProbJ[i, j] != 0:
                Entropy = Entropy - ProbJ[i,j]*np.log2(ProbJ[i,j])
    return Entropy
def myMI(T1,T2):
    ENT1=myEntropy(T1)
    ENT2=myEntropy(T2)
    ENT3=myEntropyJointe(T1, T2)
    MI = ENT1+ENT2-ENT3
    return MI
X = np.random.randint(0,100,1000)
Y =np.mod(X,10)
n=np.random.randint(0, 50,1000)
Z=X+n
#plt.scatter(X,Y, marker='o')

T1=[X,Y]
T2=[X,Z]
CorXY=myRho(T1,1000)
CorXZ=myRho(T2,1000)
##plt.scatter(X,Z, marker='o')
#print(CorXY,CorXZ)

#D=myProba(X)
RES=myProbajoint(X,Y)
EntX = myEntropy(X)
EntY = myEntropy(Y)
EntZ = myEntropy(Z)
#print("Entropie X :" , EntX)
#print("Entropie Y :" , EntY)
#print("Entropie Z :" , EntZ)
BASE2 = [2**(EntX), 2**(EntY),2**(EntZ)]
#print("2^H(X) :" , BASE2[0])
#print("2^H(Y) :" , BASE2[1])
#print("2^H(Z) :" ,BASE2[2])
EntXY = myEntropyJointe(X, Y)
EntXZ = myEntropyJointe(X, Z)


print("Entropie XY",EntXY)
print("Entropie XZ",EntXZ)
infomutXY = myMI(X, Y)
infomutXZ = myMI(X, Z)
print("Mi XY :",infomutXY)
print("Mi XZ :",infomutXZ)
#counts, bins = np.histogram(X,1000)
#plt.plot(1)
#plt.hist(bins[:-1],bins,weights=counts,color = "yellow", ec="blue")

#plt.show()











