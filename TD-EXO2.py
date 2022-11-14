import csv
import numpy as np
from myRho import *
import matplotlib.pyplot as plt
def readDat_csv(NomDuFichierCSV, nbline, nbcol):
    # Auteur P. Maurine
    # Date : 13/12/2019
    # Prend le fichier csv NomDuFichierCSV de n lignes p colonnes et retourne
    # une matrice de nxp floats
    L=[]
    Sample=np.array([],dtype=np.float)
    with open(NomDuFichierCSV,newline='') as f:
        read=csv.reader(f,delimiter=";")
        for row in read:
            L.extend(row)
    Sample=[float(i) for i in L]
    Sample=np.reshape(Sample,[nbline,nbcol])
    return (Sample)
matCSV = readDat_csv ("DonneesMeteoFrance.csv", 95, 47)
def mySturge(vect):
    size = np.size(vect)

    binsnum = 1 + np.log2(size)
    binsnum = int(np.round(binsnum + 0.5))

    etendu = np.max(vect) - np.min(vect)

    binssize = etendu / (binsnum - 1)

    mat_sorted = np.zeros([size, 2])
    mat_sorted[:, 1] = vect

    mat_bins = np.zeros([binsnum, 3])

    for i in range(int(binsnum)):
        mat_bins[i, 0] = i
        mat_bins[i, 1] = (np.min(vect) + i * binssize)
        mat_bins[i, 2] = (np.min(vect) + (i + 1) * binssize)

    for i in range(size):
        for j in range(binsnum):
            if mat_sorted[i, 1] < mat_bins[j, 2] and mat_sorted[i, 1] >= mat_bins[j, 1]:
                mat_sorted[i, 0] = mat_bins[j, 0];
    return mat_sorted[:, 0]
sturgemat = [4.78,4.75,5.48,9.03,1.41,7.31,9.85,6.05,9.84,2.31]
res10 = mySturge (sturgemat)


def myScott(vect):
    size = np.size(vect)

    moy = np.mean(vect)

    ecart_moy = vect[:] - moy

    sigma = 0

    for i in range(size):
        sigma = sigma + (ecart_moy[i] ** 2)

    sigma = np.sqrt((1 / size) * sigma)

    binssize = 3.49 * sigma * size ** (-1 / 3)

    etendu = np.max(vect) - np.min(vect)

    binsnum = etendu / binssize

    binsnum = int(np.round(binsnum + 0.5))

    mat_sorted = np.zeros([size, 2])
    mat_sorted[:, 1] = vect

    mat_bins = np.zeros([binsnum, 3])

    for i in range(int(binsnum)):
        mat_bins[i, 0] = i
        mat_bins[i, 1] = (np.min(vect) + i * binssize)
        mat_bins[i, 2] = (np.min(vect) + (i + 1) * binssize)

    for i in range(size):
        for j in range(binsnum):
            if mat_sorted[i, 1] < mat_bins[j, 2] and mat_sorted[i, 1] >= mat_bins[j, 1]:
                mat_sorted[i, 0] = mat_bins[j, 0];

    return mat_sorted[:, 0]


res11 = myScott(sturgemat)


def readDat_csv(NomDuFichierCSV, nbline, nbcol):
    # Auteur P. Maurine
    # Date : 13/12/2019
    # Prend le fichier csv NomDuFichierCSV de n lignes p colonnes et retourne
    # une matrice de nxp floats
    L = []
    Sample = np.array([], dtype=np.float)
    with open(NomDuFichierCSV, newline='') as f:
        read = csv.reader(f, delimiter=";")
        for row in read:
            L.extend(row)
    Sample = [float(i) for i in L]
    Sample = np.reshape(Sample, [nbline, nbcol])
    return (Sample)

def  myEntropy (T):
    T_P= np.zeros(len(T))
    T_P=myProba(T)
    Entropy = 0
    for i in range(np.shape(T_P)[0]):
        Entropy = Entropy -T_P[0][i]*np.log2(T_P[1][i])
    return Entropy
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

matCSV = readDat_csv("DonneesMeteoFrance.csv", 95, 47)

vectTemp = np.reshape(matCSV[1:, 1:], 94 * 46)
Annee = np.zeros(len(vectTemp), int)
for i in range(len(Annee)):
    Annee[i] = 1970 + np.mod(i, 46)

temp = np.zeros((2, len(Annee)))
temp[0, :] = Annee[:]
temp[1, :] = vectTemp[:]

H1 = myEntropy(temp[:, 1])
H2 = myEntropy(temp[:, 2])
MI22 = myMI(temp[:, 2], temp[:, 2])

print("H1 :", H1)
print("H2 :", H2)
print("MI22: ", MI22)