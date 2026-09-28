def findMean(x):
    return sum(x) / len(x)

x = [1,2,4,3,5]
y = [1,3,3,2,5]

meanx = findMean(x)
meany = findMean(y)

num = 0
den = 0

for i in range(len(x)):
    num = num + ((x[i]-meanx)*(y[i]-meany))
    den = den + pow((x[i]-meanx),2)
m = num/den
print("m = ",m)

c = round(meany-m*meanx,1)
print("c = ",c)

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[1],[2],[4],[3],[5]])
Y = np.array([[1],[3],[3],[2],[5]])

reg = LinearRegression().fit(X,Y)

print("m = ", reg.coef_)
print("c = ", reg.intercept_)