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
