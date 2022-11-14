"""
import matplotlib.pyplot as plt

y=[]
y.append(0)
x=[]
for i in range(1,101):
 x.append(i)
y.append(1)
y.append(2)
for k in  range(3,100) :
 y.append(1 - 0.1*y[k-1]-0.7*y[k-2])

plt.plot(y)
plt.show()
"""


for i in range(0 ,2):
 print(i)