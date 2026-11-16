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

xmin, xmax = X[:,0].min()-1,X[:,0].max()+1
ymin, ymax = X[:,0].min()-1,X[:,0].max()+1

xx,yy = np.meshgrid(
    np.arange(xmin,xmax,.05),
    np.arange(ymin,ymax,.05),
)

grid = np.c_[xx.ravel(),yy.ravel()]
grid_scaled = scaler.transform(grid)

Z = classifier.predict(grid_scaled)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(9,6))

plt.contourf(xx,yy,Z,alpha=.3,cmap = "viridis")

plt.scatter(
    X[:,0],
    X[:,1],
    c=Y,
    cmap = "viridis",
    edgecolors = "black",
)

plt.xlabel("Alcohol")
plt.ylabel("Color intensity")
plt.title("KNN wine classification decision boundary")

plt.show()