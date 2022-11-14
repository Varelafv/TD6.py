import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import csv
def myAnova(Matrice, pvalue_crit):
    # Initialisation :
    H = 0
    F = 0
    var_intra = 0
    var_inter = 0
    obs_moy = 0
    eff_tot = 0

    # Moyenne des classes :
    for i in range(len(Matrice)):
        obs_moy = sum(Matrice[i]) + obs_moy
        eff_tot = len(Matrice[i]) + eff_tot
        # print("Moyenne de la classe", i, "=", round(obs_moy/eff_tot,1))
        # print("Effectif total =", eff_tot, "\n")
    obs_moy = obs_moy / eff_tot
    # print("Moyenne de la moyenne des classes  =", round(obs_moy,1))
    # print("Effectif Total  =", eff_tot)

    # Variance intra (= moyenne des classes) :
    sum_var_intra = 0
    for j in range(0, len(Matrice)):
        for i in range(0, len(Matrice[j])):
            sum_var_intra = sum_var_intra + (Matrice[j][i] - np.mean(Matrice[j])) ** 2
    var_intra = (1 / eff_tot) * sum_var_intra
    # print("Variance Intra  = ", round(var_intra, 3))

    # Variance inter (= moyenne des observations) :
    sum_var_inter = 0
    for j in range(0, len(Matrice)):
        sum_var_inter = sum_var_inter + (len(Matrice[j]) * (np.mean(Matrice[j]) - obs_moy) ** 2)
    var_inter = (1 / eff_tot) * sum_var_inter
    # print("Variance Inter  = ", round(var_inter, 3))

    # Valeur de la stat F :
    # var_tot = var_intra + var_inter
    F = (var_inter / (len(Matrice) - 1)) / (var_intra / (eff_tot - len(Matrice)))
    # print("Variance Totale = ", round(var_tot, 3))
    # print("Satistique F    = ", round(F, 3))

    # Hypothèse H (=0 ou 1) :
    p_value = sc.f.cdf(F, len(Matrice) - 1, eff_tot - len(Matrice))
    # print("pValue          = ", round(p_value, 4))

    if (p_value > 1 - pvalue_crit):
        H = False
        # print("H =", H, "--> Donc on rejette H0.\n")
    else:
        H = True
        # print("H =", H, "--> Donc on valide H0.\n")

    return (H, F, var_intra, var_inter)

def readDat_csv(NomDuFichierCSV, nbline, nbcol):
    # Auteur P. Maurine
    # Date : 13/12/2019
    # Prend le fichier csv NomDuFichierCSV de n lignes p colonnes et retourne
    # une matrice de nxp floats
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
    indice = 0
    for i in range(1, len(Sample[1])):
        Temp[indice] = Sample[np.where(Sample[:, 0] == Departement), i]
        indice += 1
    return (Temp)
def Temp_An(Sample, Annee):
    Temp = np.zeros((len(Sample) - 1))
    indice = 0
    for i in range(1, len(Sample)):
        Temp[indice] = Sample[i, np.where(Sample[0, :] == Annee)]
        indice += 1
    return (Temp)
def Temp_Dept(Sample, Departement):
    Temp = np.zeros(len(Sample[1]) - 1)
    indice = 0
    for i in range(1, len(Sample[1])):
        Temp[indice] = Sample[np.where(Sample[:, 0] == Departement), i]
        indice += 1
    return (Temp)
matCSV = readDat_csv("DonneesMeteoFrance.csv", 95, 47)
Temp_Fr_2000 = Temp_An(matCSV, 2000)
# print("Températures enregistrées en France lors de l'an 2000 :\n", Temp_Fr_2000, "\n")
Temp_Fr_2005 = Temp_An(matCSV, 2005)
# print("Températures enregistrées en France lors de l'an 2005 :\n", Temp_Fr_2005, "\n")
Temp_Fr_2010 = Temp_An(matCSV, 2010)
# print("Températures enregistrées en France lors de l'an 2010 :\n", Temp_Fr_2010, "\n")

Mat_An_1 = np.zeros((3, len(Temp_Fr_2000)))
for i in range(0, len(Temp_Fr_2000)):
    Mat_An_1[0][i] = Temp_Fr_2000[i]
    Mat_An_1[1][i] = Temp_Fr_2005[i]
    Mat_An_1[2][i] = Temp_Fr_2010[i]
# print("Matrice pour 2000, 2005 et 2010 :\n", Mat_An_1, "\n")
H_An1, F_An1, var_intra_An1, var_inter_An1 = myAnova(Mat_An_1, 0.05)

print("Utilisation de la fonction myAnova (Année 2000, 2005 et 2010) :")
print("--> Hypothèse    H =", H_An1, "(On rejette H0)")
print("--> Statistique  F =", round(F_An1, 2))
print("--> Variance Intra =", round(var_intra_An1, 2))
print("--> Variance Inter =", round(var_inter_An1, 2), "\n")

# Températures en Frane entre 1970, 1975 et 1980 : ----------------------------
# Extraction des données :
Temp_Fr_1970 = Temp_An(matCSV, 1970)
# print("Températures enregistrées en France lors de l'an 1970 :\n", Temp_Fr_1970, "\n")
Temp_Fr_1975 = Temp_An(matCSV, 1975)
# print("Températures enregistrées en France lors de l'an 1975 :\n", Temp_Fr_1975, "\n")
Temp_Fr_1980 = Temp_An(matCSV, 1980)
# print("Températures enregistrées en France lors de l'an 1980 :\n", Temp_Fr_1980, "\n")

Mat_An_2 = np.zeros((3, len(Temp_Fr_2000)))
for i in range(0, len(Temp_Fr_2000)):
    Mat_An_2[0][i] = Temp_Fr_1970[i]
    Mat_An_2[1][i] = Temp_Fr_1975[i]
    Mat_An_2[2][i] = Temp_Fr_1980[i]
# print("Matrice pour 1970, 1975 et 1980 :\n", Mat_An_2, "\n")
H_An2, F_An2, var_intra_An2, var_inter_An2 = myAnova(Mat_An_2, 0.05)

print("Utilisation de la fonction myAnova (Année 1970, 1975 et 1980) :")
print("--> Hypothèse    H =", H_An2, "(On rejette H0)")
print("--> Statistique  F =", round(F_An2, 2))
print("--> Variance Intra =", round(var_intra_An2, 2))
print("--> Variance Inter =", round(var_inter_An2, 2), "\n")