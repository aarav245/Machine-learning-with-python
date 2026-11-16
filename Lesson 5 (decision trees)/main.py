import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("car.csv")
data.columns = (
    "sales",
    "maintainence",
    "doors",
    "persons",
    "boot_space",
    "safety",
    "class",
)

print(data.head())
print(data.describe())
print(data.info())

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["sales"] = labelencoder.fit_transform(data["sales"])
data["maintainence"] = labelencoder.fit_transform(data["maintainence"])
data["boot_space"] = labelencoder.fit_transform(data["boot_space"])
data["safety"] = labelencoder.fit_transform(data["safety"])
data["doors"] = labelencoder.fit_transform(data["doors"])
data["persons"] = labelencoder.fit_transform(data["persons"])
data["class"] = labelencoder.fit_transform(data["class"])

X = data[["sales","maintainence","doors","persons","boot_space","safety"]]
Y = data["class"]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=2)

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(xtrain,ytrain)

ypred = classifier.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix

matrix = confusion_matrix(ytest,ypred)
sns.heatmap(matrix,annot=True,fmt="d")
plt.title("Confusion matrix")
plt.xlabel("predicted")
plt.ylabel("actual")

plt.show()

print(classification_report(ytest,ypred))