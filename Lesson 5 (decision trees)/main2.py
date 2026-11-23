import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("student_performance.csv")

print("First 5 rows: ")
print(df.head())

print("\n Dataset information: ")
print(df.info())

print("\n Dataset description: ")
print(df.describe())

print("\n Result count")
print(df["Result"].value_counts())

sns.countplot(data=df,x="Result")

plt.title("Student pass and fail count")
plt.xlabel("Result")
plt.ylabel("# of students")

plt.show()

sns.scatterplot(data=df,x="Study_Hours",y="Previous_Marks", hue = "Result",s=100)

plt.title("Study Hours versus previous marks")
plt.show()

sns.histplot(data=df, x="Attendance",hue="Result",bins=10,multiple="stack")
plt.title("Student attendance distribution")

plt.show()
plt.figure(figsize=(8,6))

sns.heatmap(df.drop(columns=["Result"]).corr(), annot = True, cmap = "coolwarm",fmt=".2f")
plt.title("Student performance correlation heatmap")
plt.show()

X = df.drop("Result",axis=1)
Y = df["Result"]

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=42,stratify=Y)

print("\n Training records: ",len(xtrain))
print("Testing records: ",len(xtest))
model = DecisionTreeClassifier(criterion="entropy",max_depth = 4, random_state=42)
model.fit(xtrain,ytrain)
print("\n decision tree training complete!")
ypred = model.predict(xtest)

print("Actual values: ")
print(ytest.values)

print("Predicted results: ")
print(ypred)
accuracy = accuracy_score(ytest,ypred)

print("\n Model accuracy:",accuracy*100, "%")
print("\nClassification report:")
print(classification_report(ytest,ypred))

cm = confusion_matrix(ytest,ypred,labels=["Fail","Pass"])

plt.figure(figsize=(7,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail","Pass"],
    yticklabels=["Fail","Pass"],
)

plt.xlabel("Predicted result")
plt.ylabel("Actual result")

plt.title("Student performance confusion matrix")

plt.show()

plt.figure(figsize=(18,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
)

plt.title("Student performance decision tree")
plt.show()

importance = pd.Series(model.feature_importances_, index = X.columns)

importance.sort_values().plot(kind = "barh", figsize=(8,5))

plt.title("Student performance feature importance")
plt.xlabel("Importance score")

plt.xlabel("importance score")

plt.show()

new_student = pd.DataFrame([[6,80,75,1]],columns=X.columns)

prediction = model.predict(new_student)

print("\n New student details:")
print(new_student)

print("\n Predicted student result: ",prediction[0])