import numpy as np
import pandas as pd
import seaborn as sns

dataurl = "boston.txt"
rawdf = pd.read_csv(dataurl, sep =r"\s+", skiprows=22, header = None)

data = np.hstack([rawdf.values[::2,:],rawdf.values[1::2,:2]])

target = rawdf.values[1::2,2]

featurenames = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]

boston = pd.DataFrame(data, columns=featurenames)
print(boston.head())

boston["MEDV"] = target

X = boston[["LSTAT","RM"]]
Y = boston["MEDV"]

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.2,random_state=5)

from sklearn.linear_model import LinearRegression

linmodel = LinearRegression()
linmodel.fit(xtrain,ytrain)

from sklearn.metrics import mean_squared_error

ytestpred = linmodel.predict(xtest)

rmse_linmodel = np.sqrt(mean_squared_error(ytest, ytestpred))

print("The RMSE in case of multivariable regression is: ",rmse_linmodel)

from sklearn.preprocessing import PolynomialFeatures

polyfeature = PolynomialFeatures(degree = 2)

xtrainpoly = polyfeature.fit_transform(xtrain)
polymodel = LinearRegression()

polymodel.fit(xtrainpoly,ytrain)

xtestpoly = polyfeature.transform(xtest)

ytestpredpoly = polymodel.predict(xtestpoly)

rmsepolymodel = np.sqrt(mean_squared_error(ytest,ytestpredpoly))

print("The RMSE in case of polynomial regression is: ",rmsepolymodel)