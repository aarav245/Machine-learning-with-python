import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("titanic.csv")

print(data.isnull().sum())

print("Median of age column %.2f"%(data["Age"].median(skipna=True)))
print(
    "Percent of missing records in cabin %.2f"
    %((data["Cabin"].isnull().sum()/data.shape[0])*100)
)
print("Most common boarding point: %s" % data["Embarked"].value_counts().idxmax())

data["Age"].fillna(data["Age"].median(skipna = True), inplace = True)
data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(), inplace = True)
data.drop("Cabin",axis = 1,inplace=True)

print(data.isnull().sum())
data.drop("PassengerId",axis = 1,inplace=True)
data.drop("Name",axis=1,inplace=True)
data.drop("Ticket",axis=1,inplace=True)

data["TravelAlone"] = np.where((data["SibSp"]+data["Parch"])>0,0,1)

data.drop("SibSp",axis=1,inplace=True)
data.drop("Parch",axis=1,inplace=True)

print(data.head())

from sklearn import preprocessing

labelencoder = preprocessing.LabelEncoder()
data["Sex"] = labelencoder.fit_transform(data["Sex"])
data["Embarked"] = labelencoder.fit_transform(data["Embarked"])