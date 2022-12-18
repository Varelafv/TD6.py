import numpy as np
X1 = np.random.normal(0, 2, 500)
Y=[X1,X1]
Y[:][0]=1
print(Y)
teta=np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5, 10])
n=np.random.normal(0,teta**2,10000)
print(teta)