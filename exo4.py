import csv
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
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
    return (pq)
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

D=readDat_csv("DonneesMeteoFrance.csv", 95, 47)
Temp = np.reshape (D[1:,1:], 94*46)
Annee = np.zeros(len(Temp), int)
for i in range (len(Annee)):
    Annee [i] = 1970 + np.mod(i,46)

matTemp = np.zeros((2, len(Annee)))
matTemp [0, :] = Annee [:]
matTemp [1, :] = Temp [:]
print(matTemp)
PearSON = myRho(matTemp,4324)
TempSpear = mysort(matTemp)
#print(len(matTemp[0][0:]))
Spearman = myRho(TempSpear,4324)
print(PearSON)
print(Spearman)
plt.scatter(matTemp[0],matTemp[1], marker='o')
plt.title(" Nuage de points temperature et annees")
plt.xlabel("Années")
plt.ylabel("Temperature")
plt.show()