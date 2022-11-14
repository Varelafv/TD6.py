import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.stats as sc
def myTest(T1, T2, alpha):
    # Statistique de Test T :
    Moy_T1 = np.mean(T1)
    Var_T1= np.mean(T1** 2) - (Moy_T1** 2)

    Moy_T2= np.mean(T2)
    Var_T2 = np.mean(T2 ** 2) - (Moy_T2 ** 2)

    T = (Moy_T1 - Moy_T2) / (np.sqrt(Var_T1 / np.size(T1)+ Var_T2/ np.size(T2)))

    beta = sc.t.cdf(T, np.size(T1)-1) # (T, degré de liberté)0
    # Hypothèse :
    if beta >(1 - alpha / 2):
        # On est dans la zone de conformité
        # On valide H1èè
        H = 1
    else :
        # On rejette H1, et on valide H0
        H = 0
    return(T,beta,H)
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
def Temp_Dept(Sample, Departement):
    Temp = np.zeros(len(Sample[1]) - 1)
    indice = 0
    for i in range(1, len(Sample[1])):
        Temp[indice] = Sample[np.where(Sample[:, 0] == Departement), i]
        indice += 1
    return (Temp)
# Extraire les températures selon l'année :
def Temp_year(Sample, Annee):
    Temp = np.zeros((len(Sample) - 1))
    indice = 0
    for i in range(1, len(Sample)):
        Temp[indice] = Sample[i, np.where(Sample[0, :] == Annee)]
        indice += 1
    return (Temp)
Table=readDat_csv("DonneesMeteoFrance.csv", 95, 47)
print("Données : \n", Table, "\n")

Gard = 30
Temp_Gard = Temp_Dept(Table, Gard)
#print("Températures enregistrées de 1970 à 2015, pour le département n°", Gard, "(Le Gard) : \n", Temp_Gard, "\n")
Temp_moy_Gard = round(np.mean(Temp_Gard), 2)

Vilaine = 35
Temp_Vilaine = Temp_Dept(Table, Vilaine)
#print("Températures enregistrées de 1970 à 2015, pour le département n°", Vilaine, "(L'Ille-et-Vilaine) : \n",Temp_Vilaine, "\n")
Temp_moy_Vilaine = round(np.mean(Temp_Vilaine), 2)


Herault = 34
Temp_Herault = Temp_Dept(Table,Herault)

#print("Températures enregistrées de 1970 à 2015, pour le département n°", Herault, "(L'Hérault) : \n",  Temp_Herault, "\n")
Temp_moy_Herault = round(np.mean(Temp_Herault), 2)

print("Température moyenne de 1970 à 2015 :")
print("le Gard =", Temp_moy_Gard, "°C.")
print(" l'Ille-et-Vilaine =", Temp_moy_Vilaine, "°C.")
print(" l'Hérault =", Temp_moy_Herault, "°C.\n")
Gard_Vilaine = myTest( Temp_Gard, Temp_moy_Vilaine, 0.05)
Gard_Herault = myTest(Temp_Gard, Temp_Herault, 0.05)

print("--> Entre Le Gard et l'Ille-et-Vilaine : T =", round(Gard_Vilaine[0], 4), "&  H =", Gard_Vilaine[2])
print("--> Entre Le Gard et l'Hérault         : T =", round(Gard_Herault[0], 4), " &  H =", Gard_Herault[2], "\n")



Temp11 = Temp_year(Table, 2000)
print("Températures moy en Fr lors de l'an 2000 :\n", round(np.mean(Temp11), 2), "\n")
Temp22 = Temp_year(Table, 2010)
print("Température en Fr l'an 2010 :\n",round(np.mean(Temp22), 2), "\n")





Temp1 = Temp_year( Table, 1970)
print("Températures moy  l'an 1970 :\n",round(np.mean(Temp1), 2), "\n")
Temp2 = Temp_year(Table, 1980)
print("Températures moy  l'an 1980 :\n", round(np.mean(Temp2), 2), "\n")

An_1 = myTest(Temp11, Temp22, 0.05)
An_2 = myTest(Temp1, Temp2, 0.05)
print("Utilisation de la fonction myTest (Températures en France) :")
print("--> Entre 2000 et 2010 : T =", round(An_1[0], 4), "&  H =", An_1[2])
print("--> Entre 1970 et 1980 : T =", round(An_2[0], 4), "&  H =", An_2[2])