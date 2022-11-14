import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

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

def Mat_M(nb_column):
    Matrice = np.zeros((2, nb_column))

    for i in range(0, len(Matrice[0])):
        # Ligne 1 = classe
        Matrice[0][i] = np.random.uniform(0, 3)
        # Ligne 2 = observation
        Matrice[1][i] = np.random.normal(2, 10)
    return (Matrice)
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


def myEpdf(vecteur):
    n = len(vecteur)
    nbins = int(np.round(1 + np.log2(n)))
    x = np.linspace(np.min(vecteur), np.max(vecteur) + 0.1 * np.max(vecteur), nbins)
    noccurence = np.zeros(nbins-1, dtype='int')
    xaxis = np.zeros(nbins - 1, dtype='float')
    bin_width = (x[1] - x[0])
    for i in range(len(x) - 1):
        idx = np.where((vecteur >= x[i]) & (vecteur < x[i + 1]))[0]
        noccurence[i] = len(idx)/bin_width
        xaxis[i] = (x[i + 1] - x[i]) / 2 + x[i]
    noccurence = noccurence /n
    #plt.plot(xaxis, noccurence)
    mean = np.mean(xaxis)
    sigma = np.sqrt(np.var(xaxis))
    theor = sc.norm.pdf(np.linspace(-20,20,500),1,2)
    plt.plot(np.linspace(0,3,500), theor, 'r')
    return (np.linspace(-20,20,500),theor)
def studentTheorique(v,n,bins):
    moy=np.mean(v)
    sigma=np.sqrt(np.var(v))
    x=np.linspace(min(bins),max(bins),n)
    y=sc.norm.pdf(x,moy,sigma)
    return x,y
def myAnova (Matrice, pvalue_crit):
    H=0
    F=0
    moy_obs =0
    eff_tot=0
    for i in range(len(Matrice)):
        moy_obs = sum(Matrice[i]) + moy_obs
        eff_tot = len(Matrice[i]) + eff_tot
    moy_obs= moy_obs/eff_tot

    sum_var_intra = 0
    for j in range(0, len(Matrice)):
        for i in range(0, len(Matrice[j])):
            sum_var_intra = sum_var_intra + (Matrice[j][i] - np.mean(Matrice[j])) ** 2
    var_intra = (1 / eff_tot) * sum_var_intra
    #print("Variance Intra  = ", round(var_intra, 3))
    sum_var_inter = 0
    for j in range(0, len(Matrice)):
        sum_var_inter = sum_var_inter + (len(Matrice[j]) * (np.mean(Matrice[j]) - moy_obs) ** 2)
    var_inter = (1 / eff_tot) * sum_var_inter
    var_tot = var_intra + var_inter
    F = (var_inter / (len(Matrice) - 1)) / (var_intra / (eff_tot - len(Matrice)))
    p_value = sc.f.cdf(F, len(Matrice) - 1, eff_tot - len(Matrice))
    if (p_value > 1 - pvalue_crit):
        H = False
        # print("H =", H, "--> Donc on rejette H0.\n")
    else:
        H = True
        # print("H =", H, "--> Donc on valide H0.\n")

    return (H, F, var_intra, var_inter)

Notes = [[10, 11, 11, 12, 13, 13], [8, 11, 11, 13, 14, 15, 16, 16], [10, 13, 14, 14, 15, 16, 16]]
#print(" Les Matrice de test :\n", Notes, "\n")
#H, F, var_intra, var_inter=myAnova(Notes, 0.05)
"""print("H :",H)
print("F :",F)
print("var_intra :",var_intra)
print("var_inter :",var_inter)
"""
x = np.linspace(0, 15, 500)
#-------------------------------------------GRAPH1----------------------#
T_F1=[]
n = 500
for k in range(0, n):
    M = Mat_M(n)
    F = myAnova(M, 0.05)[1]
    T_F1.append(F)
#print(T_F1)
#myEpdf(T_F)
count, bins = np.histogram(T_F1, int(1 + np.log2(len(T_F1))))
plt.title(" Fonctions de densité de probabilité pour n = 500")
plt.xlabel('x')
plt.ylabel('F')
plt.figure(1)
plt.hist(bins[:-1], bins, weights=count, color='lightblue', edgecolor='white', density=True, label="MyAnova(500)")


f_Dist = sc.f.pdf(x, 3, 497)
plt.plot(x, f_Dist, 'red')




#-------------------------------------------GRAPH2----------------------#

T_F2 = []
n2 = 1000
for i in range(0, 1000):
    M2 = Mat_M(n2)
    F2 = myAnova(M2, 0.05)[1]
    T_F2.append(F2)
plt.figure(2)
plt.title(" Fonctions de densité de probabilité pour n = 1000")
count, bins = np.histogram(T_F2, int(1 + np.log2(len(T_F2))))
plt.hist(bins[:-1], bins, weights=count, color='lightblue', edgecolor='white', density=True, label="MyAnova(1000)")

f_Dist2 = sc.f.pdf(x, 6, 993)
plt.plot(x, f_Dist2, 'red', label="Loi de Fisher(6,993)")
plt.xlabel('x')
plt.ylabel('F')




#----------------------------------------GRAPH3------------------
plt.figure(3)
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


count, bins = np.histogram(tabL1, int(1 + np.log2(len(tabL1))))
plt.hist(bins[:-1], bins, weights=count, color='lightblue', edgecolor='white', density=True, label="Sigma = 1")
x_Theo1,Theo1=studentTheorique(tabL1,int(1 + np.log2(len(tabL1))),bins)
plt.plot(x_Theo1,Theo1,'y')
count, bins = np.histogram(tabL2, int(1 + np.log2(len(tabL2))))
plt.hist(bins[:-1], bins, weights=count, color='yellow', edgecolor='white', density=True, label="Sigma = 2")
x_Theo1,Theo1=studentTheorique(tabL2,int(1 + np.log2(len(tabL2))),bins)
plt.plot(x_Theo1,Theo1,'y')
count, bins = np.histogram(tabL3, int(1 + np.log2(len(tabL3))))
plt.hist(bins[:-1], bins, weights=count, color='violet', edgecolor='white', density=True, label="Sigma = 4")
x_Theo1,Theo1=studentTheorique(tabL3,int(1 + np.log2(len(tabL3))),bins)
plt.plot(x_Theo1,Theo1,'y')
plt.xlabel('x')
plt.ylabel('F')
plt.title("Figure 3 : Fonctions de densité de probabilité pour un sigma variable")
plt.legend()

#""" Fonction de densité de probabilité de Rho
count, bins = np.histogram(tabRho, int(1 + np.log2(len(tabRho))))
plt.hist(bins[:-1], bins, weights=count, color='lightblue', edgecolor='white', density=True)
x_Theo1,Theo1=studentTheorique(tabRho,int(1 + np.log2(len(tabRho))),bins)
plt.plot(x_Theo1,Theo1,'y')
plt.xlabel('x')
plt.ylabel('Rho')
plt.title("Figure 4 : Fonction de densité de probabilité de Rho (Sigma = 2)")
plt.show()








