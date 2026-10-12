import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")
print("Columns in dataset:")
print(df.columns)

df = df[["Pclass","Age","Siblings/Spouses Aboard","Parents/Children Aboard","Fare"]]

df = df.dropna()

X = df[["Pclass","Age","Siblings/Spouses Aboard","Parents/Children Aboard"]]
Y = df["Fare"]

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=42)

linearmodel = LinearRegression()
linearmodel.fit(xtrain,ytrain)
linearpred = linearmodel.predict(xtest)

linearrmse = mean_squared_error(ytest,linearpred)**.5

print("\n=============================")
print("MULTIVARIABLE REGRESSION")
print("=============================")

print("Linear regression rmse: ",linearrmse)
print("\nCoefficients: ")
for column, coefficient in zip(X.columns, linearmodel.coef_):
    print(column, ":", coefficient)
    print("\nIntercept:",linearmodel.intercept_)
    poly = PolynomialFeatures(degree = 2, include_bias=False)
    xtrainpoly = poly.fit_transform(xtrain)
    xtestpoly = poly.fit_transform(xtest)
    polymodel = LinearRegression()
    polymodel.fit(xtrainpoly,xtestpoly)
    polypred = polymodel.predict(xtestpoly)
    polyrmse = mean_squared_error(ytest,polypred)
    print("\n=============================")
    print("POLYNOMIAL REGRESSION")
    print("=============================")
    print("Polynomial regression rmse: ", polyrmse)
    print("\n=============================")
    print("MODEL COMPARISON")
    print("=============================")
    print("Linear regression rmse: ",linearrmse)
    print("Polynomial regression rmse: ", polyrmse)
    if polyrmse < linearrmse:
        print("\nPolynomial regression performed better")
    elif linearrmse < polyrmse:
        print("\nLinear Regression performed better")
    else:
        print("\nBoth models performed the same")
    comparison = pd.DataFrame(
        {
            "Actual Fare": ytest.values,
            "linear prediction": linearpred,
            "Polynomial prediction": polypred,
        }
    )