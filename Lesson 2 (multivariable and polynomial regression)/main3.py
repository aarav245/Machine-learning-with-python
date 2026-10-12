import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    "hours":[1,2,3,4,5,6,7,8,9],
    "marks":[20,28,38,50,65,82,90,95,97]
}

df = pd.DataFrame(data)

X = df[["hours"]]
Y = df["marks"]

poly = PolynomialFeatures(degree = 2)

xpoly = poly.fit_transform(X)
model = LinearRegression()

model.fit(xpoly,Y)

predictions = model.predict(xpoly)

print("predicted marks: ")
print(predictions)

plt.scatter(X,Y)
plt.plot(X,predictions)
plt.xlabel("Study hours")
plt.ylabel("Marks")
plt.title("Polynomial regression")
plt.show()