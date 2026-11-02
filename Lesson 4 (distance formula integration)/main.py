import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data.csv")

print(data.head())

X = data[["sepal_length","sepal_width","petal_length","petal_width"]]
Y = data["species"]

print(X.head())
print(Y.head())

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=5)

from sklearn.preprocessing import StandardScaler, LabelEncoder

standard_scaler = StandardScaler()

xtrain = standard_scaler.fit_transform(xtrain)
xtest = standard_scaler.transform(xtest)

labelencoder = LabelEncoder()

ytrain = labelencoder.fit_transform(ytrain)
ytest = labelencoder.transform(ytest)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(xtrain,ytrain)

ypred = classifier.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix

matrix = confusion_matrix(ytest,ypred)

sns.heatmap(matrix,annot=True,fmt = "d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(ytest,ypred,target_names=labelencoder.classes_))