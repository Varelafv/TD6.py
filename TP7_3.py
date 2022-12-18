import numpy as np
import matplotlib.pyplot as plt
from  math import *
import scipy .stats as sc
import csv


def myRegLinMult (Y,B):
    n=round(np.size(B) /np.shape(B)[0])
    X = np.ones((np.shape(B)[0],n + 1), dtype='float')
    for c in range(1,n + 1):
        X[:, c] = B[:]
    #Estimation des paramètres :
    Beta =np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),Y)
    #Estimation de Y :
    EstY=np.dot(X,Beta)
    #Le résidu
    e=Y-EstY
    N=len(Y)
    sommeN=0
    SomeD=0
    #calcule le coefficient de détermination
    #R2= 1- (sommeN/sommeD)
    # le coefficient de détermination ajusté
    R2=1-(np.sum(e**2))/np.sum((Y-np.mean(Y))**2)
    return (Beta,EstY,e,R2)
def myRegLinMult2 (Y,B):
    X = np.ones((np.shape(B)[0], np.shape(B)[1] + 1), dtype='float')
    for c in range(1, np.shape(B)[1] + 1):
        X[:, c] = B[:, c - 1]

    #Estimation des paramètres :
    Beta =np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),Y)
    #Estimation de Y :
    EstY=np.dot(X,Beta)
    #Le résidu
    e=Y-EstY
    N=len(Y)
    sommeN=0
    SomeD=0
    #calcule le coefficient de détermination
    #R2= 1- (sommeN/sommeD)
    # le coefficient de détermination ajusté
    R2=1-(np.sum(e**2))/np.sum((Y-np.mean(Y))**2)
    return (Beta,EstY,e,R2)
def readDat_csv(NomDuFichierCSV, nbline, nbcol):
    L = []
    Sample = np.array([], dtype=float)
    with open(NomDuFichierCSV, newline='') as f:
        read = csv.reader(f, delimiter=";")
        for row in read:
            L.extend(row)
    Sample = [float(i) for i in L]
    Sample = np.reshape(Sample, [nbline, nbcol])
    return (Sample)
def Temp_Dept(Sample, Departement):
    Temp = np.zeros(len(Sample[1]) - 1)
    Temp= Sample[Departement,1:]
    indice = 0
    return (Temp)
def myRegLin(X,Y):
    X=np.array(X)
    Y=np.array(Y)
    print(X)
    # Espérence :
    E_x = np.mean(X)
    E_xx =np.mean(X**2)
    E_y = np.mean(Y)
    E_yy = np.mean(Y ** 2)
    E_xy = np.mean(np.dot(X,Y))
    plt.plot(X,Y,"o")
    # Covariance :
    Cov_xy = E_xy - E_x * E_y

    # Variance :
    Var_x = E_xx - (E_x**2)
    Var_y = E_yy - (E_y** 2)
    B1= Cov_xy/Var_x
    B0= E_y - B1*np.mean(X)
    #calcul des residus
    E= Y - (B0 + B1 * X)
    V= np.mean(E**2)  -  np.mean(E)**2
    R= Cov_xy**2/(Var_x*Var_y)

    Yr=B1*X+B0
    p=sqrt(R)
    #print(R)
    #Yjuste=B0 + B1*np.exp
    plt.figure()
   # plt.plot(X,Y,'ro')
    #plt.plot(x,Yjuste,'r+')
    plt.plot(X, Yr)
    #plt.plot(np.mean(X),np.mean(Yr) ,'ro')
    #plt.figure(1)
    count, bins = np.histogram(E, 10)
    #plt.hist(bins[:-1], bins, weights=count, color="red", edgecolor="black", density=True)
    return(B0,B1,R,V)
def myRegLinMult(Y,B):
    X = np.ones((np.shape(B)[0], np.shape(B)[1] + 1), dtype='float')
    for c in range(1, np.shape(B)[1] + 1):
        X[:, c] = B[:, c - 1]

    #Estimation des paramètres :
    Beta =np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),Y)
    #Estimation de Y :
    EstY=np.dot(X,Beta)
    #Le résidu
    e=Y-EstY
    N=len(Y)
    sommeN=0
    SomeD=0
    #calcule le coefficient de détermination
    #R2= 1- (sommeN/sommeD)
    # le coefficient de détermination ajusté
    R2=1-(np.sum(e**2))/np.sum((Y-np.mean(Y))**2)
    return (Beta,EstY,e,R2)

matCSV = readDat_csv("DonneesMeteoFrance.csv", 95, 47)
Tconc=[]
T=[]
A=[]
Ttemp=[]
Anne=Temp_Dept(matCSV, 0)
print("Données : \n", T, "\n")
for i in range(1,95):
    T=Temp_Dept(matCSV, i)
    Moy = np.mean(T)
    Tconc.append(Moy)
    T=[]

print("Veleur Concatenee=",Tconc)
print(len(Tconc))
#EXO C
for i in range(1,95):
    T=Temp_Dept(matCSV, i)
    for a in range(len(T)):
        T[a] = T[a] - Tconc[i-1]
    Anne = Temp_Dept(matCSV, 0)
    A.extend(Anne)
    Ttemp.extend(T)


print(Ttemp)
Coef=myRegLin(A,Ttemp)
#CoefDeter=myRegLin(np.array(Anne,np.array(Tconc ))
plt.show()





