import numpy as np
import matplotlib.pyplot as plt
from  math import sqrt
import scipy .stats as sc

def myRegLinMult (Y,B):
    X=np.ones((np.shape(B)[0],np.shape(B)[1]+1), dtype='float')
    for c in range(1,np.shape(B)[1]+1):
       X[:,c]=B[:,c-1]
    # X[0][:]=1
    #Estimation des paramètres :
    Beta =np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),Y)
    #Estimation de Y :
    EstY=np.dot(X,Beta)
    #Le résidu
    e=Y-EstY
    N=len(Y)
    sommeN=0
    SomeD=0
    #calcule le coefficient de détermination
    #R2= 1- (sommeN/sommeD)
    # le coefficient de détermination ajusté
    R2=1-(np.sum(e**2))/np.sum((Y-np.mean(Y))**2)
    return (Beta,EstY,e,R2)

#X=np.random.uniform(0,1,10000)
#Y=np.random.uniform(0,1,10000)
X=np.random.normal(0,1,10000)
Y=np.random.normal(0,1,10000)
Z=X + np.dot(2,Y)  + np.random.normal(0,1,10000)#+2*np.random.normal(0,1,10000)
Beta,EstY,e,R2=myRegLinMult(Z,np.column_stack((X,Y)))
print("Beta \n",np.round(Beta,2))
#print("Yestime \n",EstY)
print("e  \n",e)
print("Rcarre \n ",np.round(R2,2))
teta=np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5, 10])

rcarre=np.zeros(len(teta))
for i in range(len(teta)) :
    Z=X + np.dot(2,Y) +np.random.normal(0,teta[i]**2,10000)
    rcarre[i]=myRegLinMult(Z,np.column_stack((X,Y)))[3]

plt.plot(teta,rcarre)
plt.show()
#print(np.column_stack((X,Y)))
#print(Z)



