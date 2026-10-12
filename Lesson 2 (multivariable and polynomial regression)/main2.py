import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    "size": [800,1000,1200,1500,1800,2000,2200,2500],
    "bedrooms": [2,2,3,3,4,4,5,5],
    "rooms": [4,5,6,7,8,9,10,11],
    "price": [40,50,65,80,100,115,135,155],
}

df = pd.DataFrame(data)

X = df[["size","bedrooms","rooms"]]
Y = df["price"]

xtrain, xtest, ytrain, ytest, = train_test_split(X,Y,test_size=.2,random_state=42)

model = LinearRegression()
model.fit(xtrain,ytrain)

predictions = model.predict(xtest)

rmse = mean_squared_error(ytest,predictions)**.5

print("Predicted prices: ",predictions)
print("RMSE: ",rmse)
print("Coeffecients: ", model.coef_)
print("Intercept: ", model.intercept_)