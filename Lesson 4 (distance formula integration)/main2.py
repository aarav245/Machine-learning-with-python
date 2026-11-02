import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

wine = load_wine()

X = wine.data[:,[0,9]]
Y = wine.target

print("Features: ",wine.feature_names[0],wine.feature_names[9])
print("Target names: ",wine.target_names)

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=5,stratify=Y)
scaler = StandardScaler()

xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(xtrain,ytrain)

ypred = classifier.predict(xtest)

print("Accuracy: ",accuracy_score(ytest,ypred))
matrix = confusion_matrix(ytest,ypred)

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    xticklabels=wine.target_names,
    yticklabels=wine.target_names
)

plt.title("Wine KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()