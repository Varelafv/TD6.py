import numpy as np
import csv
import matplotlib.pyplot as plt
from  math import sqrt
import scipy .stats as sc
from io import StringIO
def readDat_csv(NomDuFichierCSV, nbline, nbcol):
    # Auteur P. Maurine
    # Date : 13/12/2019
    # Prend le fichier csv NomDuFichierCSV de n lignes p colonnes et retourne
    # une matrice de nxp floats
    L=[]
    Sample=np.array([],dtype=float)
    with open(NomDuFichierCSV,newline='') as f:
        read=csv.reader(f,delimiter=";")
        for row in read:
            L.extend(row)
    Sample=[float(i) for i in L]
    Sample=np.reshape(Sample,[nbline,nbcol])
    return (Sample)
def Temp_year(Sample, Annee):
    Temp = np.zeros((len(Sample) - 1))
    indice = 0
    for i in range(1, len(Sample)):
        Temp[indice] = Sample[i, np.where(Sample[0, :] == Annee)]
        indice += 1
    return (Temp)
def interval_pre(B0,B1,alpha,n,x,y) :
    p_value = sc.t.pdf(alpha/2,n - 2)
    E = y - (B0 + B1 * x)
    V =sqrt( sum(E**2)/(n-2))

    Vx =abs( np.mean(x ** 2) - np.mean(x) ** 2)
    y1=B0+B1*x
    t=np.linspace(0,5,10000)
    T=sc.t.cdf(t,998)
    T=t[np.where(T>0.975)[0][0]]
    Ex=np.mean(x)
    D=(Ex-x)**2
    y2 = B0 + B1 * x + T*V*sqrt(1 + 1/n + D[2]/((n-1)*Vx))
    y3 = B0 + B1 * x - T * V * sqrt(1 + 1 / n + D[2] / ((n - 1) * Vx))
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x, y3)
    return (y1)
def myRegLin(X,Y ):
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
    B1= Cov_xy/Var_x
    B0= E_y - B1*np.mean(X)
    print(B0)
    print(B1)
    #calcul des residus
    E= Y - (B0 + B1 * X)
    V= np.mean(E**2)  -  np.mean(E)**2
    R= Cov_xy**2/(Var_x*Var_y)
    Yr=B1*X+B0
    p=sqrt(R)
    #Yjuste=B0 + B1*np.exp(X)
    plt.plot(X,Y,'bo')
    #plt.plot(x,Yjuste,'r+')
    plt.plot(x, Yr),
    #plt.plot(np.mean(X),np.mean(Yr) ,'ro')
    #plt.figure(1)
    count, bins = np.histogram(E, 10)
    #plt.hist(bins[:-1], bins, weights=count, color="red", edgecolor="black", density=True)
    return(B0,B1,R,V)
#Y=np.random.uniform(0,10,1000),
#x=np.random.randint(0,11,1000)

#
#B0,B1,R,V=myRegLin(x,y)
#alpha=0.05

#y1=interval_pre(B0,B1,alpha,n,x,y)
Herault= np.loadtxt('herault.txt',dtype= 'float',delimiter=";" )
x=Herault[1][:]
print()
Table=readDat_csv("DonneesMeteoFrance.csv", 95, 47)
#x=Temp_year(Table,1970)
n=np.random.normal(0,3,len(x))
y=2*x+3 + n
B0,B1,R,V=myRegLin(x,y)

plt.show()