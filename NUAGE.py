import numpy as np
GR=0
X=np.random.randint(0,15,10)
#X.sort()
GRAX=np.zeros(np.shape(T),dtype='float')
Tu=np.unique(X)
print(X)
print(Tu)
for i in Tu :
    iNd = np.where(X == i)[0]
    print(iNd)
    nx = len(iNd)
    print(nx)
    GRAX[iNd] = np.mean(np.arange(GR, GR + nx))
    GR = GR + nx
    print(GRAX)

