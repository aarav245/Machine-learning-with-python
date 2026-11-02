import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = load_breast_cancer()
X = pd.DataFrame(data.data,columns=data.feature_names)
Y = data.target
print("Dataset shape:",X.shape)
print("\n First 5 rows:")
print(X.head())

print("\nMissing Values: ", X.isnull().sum().sum())

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=42,stratify=Y)
model = make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
print("\nAccuracy: ")
print(accuracy_score(ytest,ypred))
print("\nConfusion matrix: ")
print(confusion_matrix(ytest,ypred))
print("\nClassification report: ")
print(classification_report(ytest,ypred,target_names=data.target_names))

newexample = xtest.iloc[[0]]

prediction = model.predict(newexample)[0]
probabilities = model.predict_proba(newexample)[0]

print("\nNew Example Prediction: ")
print("Predicted class: ",data.target_names[prediction])
print("Class probabilities: ")

for classname, probability in zip(data.target_names,probabilities):
    print(f"{classname}:{probability:.2%}")
