import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("titanic.csv")

print("Dataset shape:", data.shape)
print("\nMissing values")
print(data.isnull().sum())

print("Median of age column: %.2f" % data["Age"].median())

print("Missing age percentage:%.2f%%" % (data["Age"].isnull().sum()/len(data)*100))

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

print("\nMissing values after preprocsessing:")
print(data.isnull().sum())

data["TravelAlone"] = np.where(
    (data["Siblings/Spouses Aboard"]+data["Parents/Children Aboard"]) > 0,0,1
    
)
data.drop(columns = ["Siblings/Spouses Aboard","Parents/Children Aboard"],inplace = True)

data["Sex"] = data["Sex"].map({"male":0,"female":1})

print("\nProcessed data")
print(data.head())

X = data[["Pclass","Sex","Age","Fare","TravelAlone"]]
Y = data["Survived"]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=2,stratify=Y)

print("\nTraining records: ",len(xtrain))
print("Testing records: ",len(xtest))

from sklearn.linear_model import LogisticRegression

linmodel = LogisticRegression(max_iter=1000)
linmodel.fit(xtrain,ytrain)

ypred = linmodel.predict(xtest)
print("\nPredicted values: ")
print(ypred)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

matrix = confusion_matrix(ytest,ypred)

print("\nConfusion Matrix: ")
print(matrix)

sns.heatmap(matrix,annot=True,fmt="d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n Classification report: ")
print(classification_report(ytest,ypred))

print("Accuracy: %.2f%%" % (accuracy_score(ytest,ypred)*100))