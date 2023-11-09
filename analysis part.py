from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
import numpy as np
from sklearn.metrics import confusion_matrix


heartdata = pd.read_csv('cleveland_preprocessed.csv')
print("All Data Set")
print(heartdata)
heartdata.info()
heartdata.describe()

#heartdata.hist()
Heart_disease_person = heartdata.loc[heartdata['Heart_disease_label'] == 1]
No_Heart_disease_person = heartdata.loc[heartdata['Heart_disease_label'] == 0]
print("Heart_disease_person: ", len(Heart_disease_person), " ", "No_Heart_disease_person: ", len(No_Heart_disease_person))  

print(Heart_disease_person)
print(No_Heart_disease_person)

X = heartdata.drop('Heart_disease_label', axis=1)  # read features, drop "Class" column
y = heartdata['Heart_disease_label']   # label column
print(X); #All Data
print(y);
X = heartdata.drop('Heart_disease_label', axis=1)
#X = heartdata.drop(columns = ['age','chol','fbs','restecg','exang','thal'],axis= 1)
heartdata.Heart_disease_label.value_counts()

plt.bar(heartdata['Heart_disease_label'].unique(), heartdata['Heart_disease_label'].value_counts(), color = ['green', 'red'], width = 0.3)
plt.xticks([0, 1])
plt.xlabel('Heart_disease_label')
plt.ylabel('Count')
plt.title('Count of each Heart_disease_label')
plt.show()


plt.scatter(x=heartdata.age[heartdata.Heart_disease_label==1], y=heartdata.thalach[(heartdata.Heart_disease_label==1)], c="red")
plt.scatter(x=heartdata.age[heartdata.Heart_disease_label==0], y=heartdata.thalach[(heartdata.Heart_disease_label==0)], c = 'green')
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

countNoDisease = len(heartdata[heartdata.Heart_disease_label == 0])
countHaveDisease = len(heartdata[heartdata.Heart_disease_label == 1])
print("Percentage of Persons Having No Heart Disease: {:.2f}%".format((countNoDisease / (len(heartdata.Heart_disease_label))*100)))
print("Percentage of Persons Having Heart Disease: {:.2f}%".format((countHaveDisease / (len(heartdata.Heart_disease_label))*100)))
sns.countplot(x='sex', data=heartdata, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

pd.crosstab(heartdata.sex,heartdata.Heart_disease_label).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["No Heart Disease", "Have Heart Disease"])
plt.ylabel('Frequency')
plt.show()

pd.crosstab(heartdata.age,heartdata.Heart_disease_label).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()

pd.crosstab(heartdata.slope,heartdata.Heart_disease_label).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()

pd.crosstab(heartdata.fbs,heartdata.Heart_disease_label).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Have no heart Disease", "Have heart Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()

pd.crosstab(heartdata.cp,heartdata.Heart_disease_label).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()

#####Creating Dummy Variables

a = pd.get_dummies(heartdata['cp'], prefix = "cp")
b = pd.get_dummies(heartdata['thal'], prefix = "thal")
c = pd.get_dummies(heartdata['slope'], prefix = "slope")

frames = [heartdata, a, b, c]
heartdata = pd.concat(frames, axis = 1)
heartdata.head()
heartdata = heartdata.drop(columns = ['cp', 'thal', 'slope'])
heartdata.head()
#print("heartdata" , heartdata.head())
