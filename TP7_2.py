import numpy as np
import matplotlib.pyplot as plt
from  math import *
import scipy .stats as sc

def myRegLinMult (Y,B):
    n=round(np.size(B) /np.shape(B)[0])
    X = np.ones((np.shape(B)[0],n + 1), dtype='float')
    for c in range(1,n + 1):
        X[:, c] = B[:]
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
def myRegLinMult2 (Y,B):
    X = np.ones((np.shape(B)[0], np.shape(B)[1] + 1), dtype='float')
    for c in range(1, np.shape(B)[1] + 1):
        X[:, c] = B[:, c - 1]

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
x= np.load('X-TD7-EX2.npy', mmap_mode='r')
y= np.load('Y-TD7-EX2.npy', mmap_mode='r')
C1=myRegLinMult(y,x)[3]
print("R1= ",C1)
C2=myRegLinMult2(y,np.column_stack((x,x**2)))[3]
print("R2= ",C2)
C3=myRegLinMult2(y,np.column_stack((x,x**2,x**3)))[3]
print("R3= ",C3)
C4=myRegLinMult2(y,np.column_stack((x,x**2,x**3,x**4)))[3]
print("R4= ",C4)
C5=myRegLinMult2(y,np.column_stack((x,x**2,x**3,x**4,x**5)))[3]
print("R5= ",C5)
C6=myRegLinMult2(y,np.column_stack((x,x**2,x**3,x**4,x**5,x**6)))[3]
print("R6= ",C6)
C7=myRegLinMult2(y,np.column_stack((x,x**2,x**3,x**4,x**5,x**6,x**7)))[3]
print("R7= ",C7)
C8=myRegLinMult2(y,np.column_stack((x,x**2,x**3,x**4,x**5,x**6,x**7,x**8)))[3]
print("R8= ",C8)





print(C1)
#plt.plot(teta,rcarre)
plt.show()
#print(np.column_stack((X,Y)))




