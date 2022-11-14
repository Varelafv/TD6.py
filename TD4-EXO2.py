import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
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
    plt.plot(xaxis, noccurence)
    mean = np.mean(xaxis)
    sigma = np.sqrt(np.var(xaxis))
    theor = sc.norm.pdf(np.linspace(-20,20,500),1,2)
   # plt.plot(np.linspace(0,2,500), theor, 'r')
    return (np.linspace(-20,20,500),theor)
def myTest(T1, T2, alpha):
    # Statistique de Test T :
    Moy_T1 = np.mean(T1)
    Var_T1= np.mean(T1** 2) - (Moy_T1** 2)

    Moy_T2= np.mean(T2)
    Var_T2 = np.mean(T2 ** 2) - (Moy_T2 ** 2)

    T = (Moy_T1 - Moy_T2) / (np.sqrt(Var_T1 / len(T1)+ Var_T2/len(T2)))

    beta = sc.t.cdf(T,len(T1)-1) # (T, degré de liberté)0
    # Hypothèse :
    if beta >(1 - alpha / 2):
        # On est dans la zone de conformité
        # On valide H1èè
        H = 1
    else :
        # On rejette H1, et on valide H0
        H = 0
    return(T,beta,H)
def studentTheorique(v,n):
    moy=np.mean(v)
    sigma=np.sqrt(np.var(v))
    x=np.linspace(-10,10,n)
    y=sc.norm.pdf(x,moy,sigma)
    return x,y
cont = 500
T_1 = np.zeros(cont, dtype=float)
H_1 = np.zeros(cont, dtype=int)
"""T_2 = np.zeros(i, dtype=float)
H_2 = np.zeros(i, dtype=int)"""
"""T_3 = np.zeros(i, dtype=float)
H_3 = np.zeros(i, dtype=int)"""

for i in range(cont):
    n1 = 500
    n2 = 500
    Sample_1 = np.random.normal(0, 2, n1)
    Sample_2 = np.random.normal(0, 2, n2)
    T_1[i], pvalue_T_1, H_1[i] = myTest(Sample_1, Sample_2, 0.05)
    #print(T_1[i], pvalue_T_1, H_1[i])

x1,T2=myEpdf(T_1)
count,bins=np.histogram(T_1,10)
plt.hist(bins[:-1], bins,weights=count,color="red",edgecolor="black",density=True)
x_Theo1,Theo1=studentTheorique(T_1,500)
plt.plot(x_Theo1,Theo1,'y')
for i in range(cont):
    n1 = 500
    n2 = 500
    Sample_1 = np.random.normal(0, 2, n1)
    Sample_2 = np.random.normal(0.5, 2, n2)
    T_1[i], pvalue_T_1, H_1[i] = myTest(Sample_1, Sample_2, 0.05)
    #print(T_1[i], pvalue_T_1, H_1[i])

x1,T2=myEpdf(T_1)
count,bins=np.histogram(T_1,10)
plt.hist(bins[:-1], bins,weights=count,color="red",edgecolor="black",density=True)

for i in range(cont):
    n1 = 500
    n2 = 500
    Sample_1 = np.random.normal(0, 2, n1)
    Sample_2 = np.random.normal(-0.5, 2, n2)
    T_1[i], pvalue_T_1, H_1[i] = myTest(Sample_1, Sample_2, 0.05)
    #print(T_1[i], pvalue_T_1, H_1[i])

x1,T2=myEpdf(T_1)
count,bins=np.histogram(T_1,10)
plt.hist(bins[:-1],bins,weights=count,color="red",edgecolor="black",density=True)
#plt.plot(x1,T2)

x_Theo1,Theo1=studentTheorique(T_1,500)
plt.plot(x_Theo1,Theo1,'r')
plt.show()