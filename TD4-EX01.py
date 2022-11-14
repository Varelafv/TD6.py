import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import math

def myEpdf(vecteur, figure):
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
    plt.plot(xaxis, noccurence,'k+')
    mean = np.mean(xaxis)
    sigma = np.sqrt(np.var(xaxis))
    y=theor = sc.norm.pdf(np.linspace(-10,10,1000),1,2)
    plt.plot(np.linspace(-10,10,1000), theor, 'r')
    return 0

def myEcdf(X) :
    nbBins = int(np.round(1+np.log2(len(X))))
    Xmax = np.max(X)
    Xmin = np.min(X)
    x = np.linspace(Xmin, Xmax+0.1*Xmax, nbBins)
    x_axis = np.zeros(nbBins-1, dtype='float')
    noccurence = np.zeros(nbBins-1, dtype = 'int')

    for i in range(len(x)-1) :
        idx = np.where((X<x[i]))[0]
        noccurence[i] = len(idx)
        x_axis[i] = (x[i])
    print(x_axis)

    noccurence = noccurence/len(X)
    print(noccurence)

    return(noccurence, x_axis)

n = 100000
x_axis = np.random.normal(1, 2, n)
myEpdf(x_axis,2)
cdf, x2 = myEcdf(x_axis)
plt.figure(2)
plt.plot(x2, cdf)
y = sc.norm.cdf(x2,np.mean(x_axis), np.sqrt(np.var(x_axis)))
plt.plot(x2, y, 'g+')
plt.show()
