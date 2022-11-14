from math import sqrt
def myRho(T,n):
    COV=0;
    pq =0 ;
    SY=0;
    SX=0;
    EX=sum(T[0][0:])/float(len(T[0][0:])) #La experence de x
    EY=sum(T[1][0:])/float(len(T[1][0:])) #La experence de y
    for i in range(n):
        COV = COV + (T[0][i] - EX) * (T[1][i] - EY)
        SX = SX + (T[0][i] - EX) ** 2
        SY = SY + (T[1][i] - EY) ** 2
    pq = COV /sqrt(SX * SY) #La correlation
    COV = COV / n #La covolution de x y
    t=pq/sqrt((1-pq**2)/(n-2)) #La statistique sur la correlation de x y
    return (pq)
"""
T=[[0,2,4,2,2,1,4,1,4,4],[8,6,14,9,15,10,15,4,18,19]]
COV,pq,t=myRho(T,10)
print("La Covariance {:.2f}".format(COV))
print("La Corrélation de Pearson  {:.2f}".format(pq))
print("La Statistique T {:.2f}".format(t))
"""
