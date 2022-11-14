

# Alias :
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.stats import t  # Importer loi de Student
from scipy.stats import ttest_ind as ttest
import csv


# ----------------------------- FONCTIONS -------------------------------#
# Matrice my Anova -----------------------------------------------------#
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


# Covariance, Corrélation & Statistique T ------------------------------#
def myRho(X, Y):
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

    # Ecart-type :
    Ecart_x = np.sqrt(Var_x)
    Ecart_y = np.sqrt(Var_y)

    # Correlation :
    Rho_xy = Cov_xy / (Ecart_x * Ecart_y)

    # Statistique T :
    T = Rho_xy / np.sqrt((1 - Rho_xy ** 2) / (len(X) - 2))

    return (Cov_xy, Rho_xy, T, Ecart_x, Ecart_y)


# Remplir Matrice M  --------------------------------------------------#
def Mat_M(nb_column):
    Matrice = np.zeros((2, nb_column))

    for i in range(0, len(Matrice[0])):
        # Ligne 1 = classe
        Matrice[0][i] = np.random.uniform(0, 3)
        # Ligne 2 = observation
        Matrice[1][i] = np.random.normal(2, 10)

    return (Matrice)


# Remplir Matrice L  --------------------------------------------------#
def Mat_L(nb_column, sigma):
    Matrice = np.zeros((2, nb_column))
    eta = 0

    for i in range(0, len(Matrice[0])):
        # Ligne 1 = classe
        Matrice[0][i] = np.random.uniform(0, 3)
        # Ligne 2 = observation
        eta = np.random.normal(0, sigma)
        Matrice[1][i] = np.sin((2 * np.pi * ((Matrice[0][i]))) / 6) + eta

    return (Matrice)


# Fonction Lecture d'Excel ----------------------------------------------#
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


# Extraire les températures d'un seul département : --------------------#
def Temp_Dept(Sample, Departement):
    Temp = np.zeros(len(Sample[1]) - 1)
    indice = 0
    for i in range(1, len(Sample[1])):
        Temp[indice] = Sample[np.where(Sample[:, 0] == Departement), i]
        indice += 1
    return (Temp)


# Extraire les températures selon l'année : ----------------------------#
def Temp_An(Sample, Annee):
    Temp = np.zeros((len(Sample) - 1))
    indice = 0
    for i in range(1, len(Sample)):
        Temp[indice] = Sample[i, np.where(Sample[0, :] == Annee)]
        indice += 1
    return (Temp)


# ------------------------------- APPEL ---------------------------------#
print("\n------------------------- EXERCICE 1 ------------------------- \n")
# Exercice du cours (Notes/jour) :
Notes = [[10, 11, 11, 12, 13, 13], [8, 11, 11, 13, 14, 15, 16, 16], [10, 13, 14, 14, 15, 16, 16]]
print("Matrice de test :\n", Notes, "\n")
 myAnova(Notes, 0.05)

x = np.linspace(0, 15, 1000)

# Matrice M : -----------------------------------------------------------------
tabF = []
n = 500
for k in range(0, n):
    M = Mat_M(n)
    F = myAnova(M, 0.05)[1]
    tabF.append(F)
    print("Itération", k)
count, bins = np.histogram(tabF, int(1 + np.log2(len(tabF))))
plt.hist(bins[:-1], bins, weights=count, color='lightblue', edgecolor='white', density=True, label="MyAnova(500)")

f_Dist = sc.f.pdf(x, 3, 497)
plt.plot(x, f_Dist, 'red', label="Loi de Fisher(3,497)")

plt.xlabel('x')
plt.ylabel('F')
plt.title("Figure 1 : Fonctions de densité de probabilité pour n = 500")
plt.legend()
plt.show()


# Matrice M2 : ----------------------------------------------------------------
tabF2 = []
n2 = 1000
for l in range(0, 500):
    M2 = Mat_M(n2)
    F2 = myAnova(M2, 0.05)[1]
    tabF2.append(F2)
    print("Itération", l)
count, bins = np.histogram(tabF, int(1 + np.log2(len(tabF))))
plt.hist(bins[:-1], bins, weights=count, color='lightblue', edgecolor='white', density=True, label="MyAnova(1000)")

f_Dist2 = sc.f.pdf(x, 6, 993)
plt.plot(x, f_Dist2, 'red', label="Loi de Fisher(6,993)")

plt.xlabel('x')
plt.ylabel('F')
plt.title("Figure 2 : Fonctions de densité de probabilité pour n = 1000")
plt.legend()
plt.show()


# Matrice L : -----------------------------------------------------------------
nl = 500
sigma1 = 1
sigma2 = 2
sigma3 = 4

tabL1 = []
tabL2 = []
tabL3 = []
tabRho = []

for i in range(0, nl):
    L1 = Mat_L(nl, sigma1)
    L2 = Mat_L(nl, sigma2)
    L3 = Mat_L(nl, sigma3)

    FL1 = myAnova(L1, 0.05)[1]
    FL2 = myAnova(L2, 0.05)[1]
    FL3 = myAnova(L3, 0.05)[1]

    tabL1.append(FL1)
    tabL2.append(FL2)
    tabL3.append(FL3)

    rho = myRho(L2[0], L2[1])[1]
    tabRho.append(rho)

    print("Itération", i)

count, bins = np.histogram(tabL1, int(1 + np.log2(len(tabL1))))
plt.hist(bins[:-1], bins, weights=count, color='lightblue', edgecolor='white', density=True, label="Sigma = 1")
count, bins = np.histogram(tabL2, int(1 + np.log2(len(tabL2))))
plt.hist(bins[:-1], bins, weights=count, color='yellow', edgecolor='white', density=True, label="Sigma = 2")
count, bins = np.histogram(tabL3, int(1 + np.log2(len(tabL3))))
plt.hist(bins[:-1], bins, weights=count, color='violet', edgecolor='white', density=True, label="Sigma = 4")

plt.xlabel('x')
plt.ylabel('F')
plt.title("Figure 3 : Fonctions de densité de probabilité pour un sigma variable")
plt.legend()
plt.show()

# Fonction de densité de probabilité de Rho
count, bins = np.histogram(tabRho, int(1 + np.log2(len(tabRho))))
plt.hist(bins[:-1], bins, weights=count, color='lightblue', edgecolor='white', density=True)

plt.xlabel('x')
plt.ylabel('Rho')
plt.title("Figure 4 : Fonction de densité de probabilité de Rho (Sigma = 2)")
plt.show()

plt.close('all')

print("\n------------------------- EXERCICE 2 ------------------------- \n")
# Lecture des Données Météo de France (95 Lignes & 47 Colonnes) :
matCSV = readDat_csv("DonnéesMéteoFrance.csv", 95, 47)
print("Données : \n", matCSV, "\n")

# Température moyenne (Le Gard, l'Hérault, le Rhone) : ------------------------
Dept_Gard = 30
Temp_Gard = Temp_Dept(matCSV, Dept_Gard)
# print("Températures enregistrées de 1970 à 2015, pour le département n°", Dept_Gard, "(Le Gard) : \n", Temp_Gard, "\n")

Dept_Herault = 34
Temp_Herault = Temp_Dept(matCSV, Dept_Herault)
# print("Températures enregistrées de 1970 à 2015, pour le département n°", Dept_Herault, "(L'Hérault) : \n", Temp_Herault, "\n")

Dept_Rhone = 13
Temp_Rhone = Temp_Dept(matCSV, Dept_Rhone)
# print("Températures enregistrées de 1970 à 2015, pour le département n°", Dept_Rhone, "(L'Hérault) : \n", Temp_Herault, "\n")

# Vérification des écarts des moyennes (myAnova) :
Mat_Dept = np.zeros((3, len(Temp_Gard)))
for i in range(0, len(Temp_Gard)):
    Mat_Dept[0][i] = Temp_Gard[i]
    Mat_Dept[1][i] = Temp_Herault[i]
    Mat_Dept[2][i] = Temp_Rhone[i]
# print("Matrice pour les trois département :\n", Mat_Dept, "\n")
H_Dept, F_Dept, var_intra_Dept, var_inter_Dept = myAnova(Mat_Dept, 0.05)

print("Utilisation de la fonction myAnova (Le Gard, l'Hérault et le Rhone) :")
print("--> Hypothèse    H =", H_Dept, "(On rejette H0)")
print("--> Statistique  F =", round(F_Dept, 2))
print("--> Variance Intra =", round(var_intra_Dept, 2))
print("--> Variance Inter =", round(var_inter_Dept, 2), "\n")

# Températures en Frane entre 2000, 2005 et 2010 : ----------------------------
# Extraction des données :
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