import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data = {
    "Age":[16,17,16,18,17,16,18,17,16,18,17,16,18,17,16,18,17,16,18,17],
    "Studyhours": [1,5,4,2,6,3,1,5,4,2,6,3,1,5,4,2,6,3,1,5],
    "Attendance": [40,90,80,45,95,70,35,85,75,50,95,65,30,90,80,40,98,70,35,88],
    "Previousmarks": [35,85,75,40,95,65,30,90,80,45,92,60,25,88,78,38,96,68,32,87],
    "Gender": ["Male","Female","Female","Male","Male","Female","Male","Female","Female","Female","Male","Male","Female","Male","Female","Female","Male","Male","Female","Male","Female"],
    "Result": ["Fail","Pass","Pass","Fail","Pass","Pass","Fail","Pass","Pass","Fail","Pass","Pass","Fail","Pass","Pass","Fail","Pass","Pass","Fail","Pass"],

}

df = pd.DataFrame(data)
