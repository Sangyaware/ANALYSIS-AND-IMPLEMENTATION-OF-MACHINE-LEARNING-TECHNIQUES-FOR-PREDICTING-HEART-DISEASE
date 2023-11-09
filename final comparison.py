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
from sklearn.metrics import accuracy_score


heartdata = pd.read_csv('cleveland_preprocessed.csv')
X = heartdata.drop('Heart_disease_label', axis=1)  # read features, drop "Class" column
y = heartdata['Heart_disease_label']   # label column
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

### to handle incompatible data types. Strings to numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

#######Standardize features by removing the mean and scaling to unit variance, normalizing the data.
scaler = preprocessing.StandardScaler()
X1 = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.40)

svclassifier = SVC(kernel= 'rbf')   
svclassifier.fit(X_train, y_train)   
y_pred = svclassifier.predict(X_test) 
print("Accuracy of SVM Algorithm: {:.2f}%".format(svclassifier.score(X_test,y_test)*100))
svm = svclassifier.score(X_test, y_test)*100

# train model
scaler.fit_transform(X_train,y_train)
fit_accuracy = svclassifier.score(X_train, y_train)
test_accuracy = svclassifier.score(X_test, y_test)
print(f"Train accuracy: {fit_accuracy:0.2%}")
print(f"Test accuracy: {test_accuracy:0.2%}")
####confusion matrix
svclassifier_cm=confusion_matrix(y_test,y_pred)
print(svclassifier_cm)

#precision score
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("Precision: ",precision)

#recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print("Recall is: ",recall)

#f score
print("f1 score is:",(2*precision*recall)/(precision+recall))


sns.heatmap(svclassifier_cm,annot = True, fmt = "d")
CM = pd.crosstab(y_test, y_pred)
CM


TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
fnr = FN*100/(FN+TP)
fnr

plt.show()

#### random forest
def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
    
    """
    Fit the chosen model and print out the score.
    
    """
    
    # instantiate model
    model = classifier(**kwargs)
    
    # train model
    model.fit(X_train,y_train)
    
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Train accuracy: {fit_accuracy:0.2%}")
    print(f"Test accuracy: {test_accuracy:0.2%}")
    
    return model

from sklearn.ensemble import RandomForestClassifier
randfor = RandomForestClassifier(n_estimators=100, random_state=0)
randfor.fit(X_train, y_train)
y_pred_rf = randfor.predict(X_test)

#print(y_pred_rf)
from sklearn.model_selection import learning_curve
# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X_train, y_train,
                                                        # Number of folds in cross-validation cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


score_rf = round(accuracy_score(y_pred_rf,y_test)*100,2)
print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")


#Random forest with 100 trees
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))



#Now, let us prune the depth of trees and check the accuracy.


rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, y_test)))

####confusion matrix
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(y_test, y_pred_rf)
print(matrix)
sns.heatmap(matrix,annot = True, fmt = "d")

#precision score
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred_rf)
print("Precision: ",precision)

#recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred_rf)
print("Recall is: ",recall)



#F score
f1_score = ((2*precision*recall)/(precision+recall))
print ("F1_score is : ",f1_score)


CM =pd.crosstab(y_test, y_pred_rf)
CM



TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]

#False negative rate of the model


fnr=FN*100/(FN+TP)
fnr

plt.show()


###### knn
def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
    
    """
    Fit the chosen model and print out the score.
    
    """
    
    # instantiate model
    model = classifier(**kwargs)
    
    # train model
    model.fit(X_train,y_train)
    
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Train accuracy: {fit_accuracy:0.2%}")
    print(f"Test accuracy: {test_accuracy:0.2%}")
    
    return model

from sklearn.neighbors import KNeighborsClassifier
knn = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, n_neighbors=8)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
#print(y_pred_knn)
score_knn = round(accuracy_score(y_pred_knn,y_test)*100,2)
print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")
# KNN
from sklearn.neighbors import KNeighborsClassifier
model = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier)
# Seek optimal 'n_neighbours' parameter
for i in range(1,10):
    print("n_neigbors = "+str(i))
    train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, n_neighbors=i)

n_neigbors = 1
n_neigbors = 2
n_neigbors = 3
n_neigbors = 4
n_neigbors = 5
n_neigbors = 6
n_neigbors = 7
n_neigbors = 8
n_neigbors = 9

#Confusion matrix
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(y_test, y_pred_knn)
sns.heatmap(matrix,annot = True, fmt = "d")
#precision score
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred_knn)
print("Precision: ",precision)
#recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred_knn)
print("Recall is: ",recall)
#f score
F1_score = ((2*precision*recall)/(precision+recall))
print("f1_score is : ",F1_score)

CM = pd.crosstab(y_test, y_pred_knn)
CM

TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]

#false negative rate of the model


fnr = FN*100/(FN+TP)
fnr


#false negative rate


CM = pd.crosstab(y_test, y_pred_knn)
TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
fnr = FN*100/(FN+TP)
fnr

plt.show()

#### decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
print("Accuracy using Decision Tree: {:.2f}%".format(dt.score(X_test, y_test)*100))
    
fit_accuracy = dt.score(X_train, y_train)
test_accuracy = dt.score(X_test, y_test)
print(f"Train accuracy: {fit_accuracy:0.2%}")
print(f"Test accuracy: {test_accuracy:0.2%}")

tree1 = DecisionTreeClassifier(max_depth=3, random_state=0)
tree1.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree1.score(X_test, y_test)))
dt = dt.score(X_test, y_test)*100

##### confusion matrix
dt_cm=confusion_matrix(y_test,y_pred)
sns.heatmap(dt_cm,annot = True, fmt = "d")
print(dt_cm)

#precision score
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("Precision: ",precision)

#recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print("Recall is: ",recall)

#f score
print("f1 score is :" ,(2*precision*recall)/(precision+recall))

CM = pd.crosstab(y_test, y_pred)
CM


TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
fnr = FN*100/(FN+TP)
fnr

plt.show()

##### naive bayes
def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
    
    """
    Fit the chosen model and print out the score.
    
    """
    
    # instantiate model
    model = classifier(**kwargs)
    
    # train model
    model.fit(X_train,y_train)
    
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Train accuracy: {fit_accuracy:0.2%}")
    print(f"Test accuracy: {test_accuracy:0.2%}")
    
    return model

from sklearn.naive_bayes import GaussianNB
nb = train_model(X_train, y_train, X_test, y_test, GaussianNB)
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
#print(y_pred_nb)
score_nb = round(accuracy_score(y_pred_nb,y_test)*100,2)
print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = train_model(X_train, y_train, X_test, y_test, GaussianNB)

#confusion matrix of Naive Bayes
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(y_test, y_pred_nb)
sns.heatmap(matrix,annot = True, fmt = "d")
print (matrix)
#precision score
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred_nb)
print("Precision: ",precision)

#recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred_nb)
print("Recall is: ",recall)

#f score
f1_score = ((2*precision*recall)/(precision+recall))
print ("F1_Score is : ", f1_score)


CM = pd.crosstab(y_test, y_pred_nb)
CM

TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]

#false negative rate of the model


fnr = FN*100/(FN+TP)
fnr

plt.show()
#### logistic regression

def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
    
    """
    Fit the chosen model and print out the score.
    
    """
    
    # instantiate model
    model = classifier(**kwargs)
    
    # train model
    model.fit(X_train,y_train)
    
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Train accuracy: {fit_accuracy:0.2%}")
    print(f"Test accuracy: {test_accuracy:0.2%}")
    
    return model

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
#print(y_pred_lr)
score_lr = round(accuracy_score(y_pred_lr,y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = train_model(X_train, y_train, X_test, y_test, LogisticRegression)

#Logistic Regression supports only solvers in ['liblinear', 'newton-cg'<-93.44, 'lbfgs'<-91.8, 'sag'<-72.13, 'saga'<-72.13]
clf = LogisticRegression(random_state=0, solver='newton-cg',multi_class='multinomial').fit(X_test, y_test)
#The solver for weight optimization.
#'lbfgs' is an optimizer in the family of quasi-Newton methods.
clf.score(X_test, y_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(y_test, y_pred_lr)
print (matrix)
sns.heatmap(matrix,annot = True, fmt = "d")

#precision Score
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred_lr)
print("Precision: ",precision)

#Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred_lr)
print("Recall is: ",recall)

#F-Score
f1_score = ((2*precision*recall)/(precision+recall))
print ("F1 score is : ",f1_score)



CM =pd.crosstab(y_test, y_pred_lr)

TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]

#false negative


fnr=FN*100/(FN+TP)

plt.show()

left = [1,2,3,4,5,6]   
# heights of bars 
#height = [85.24 , 84.42]
height = [svm,score_rf,score_knn,dt,score_nb,score_lr]

# labels for bars 
tick_label = ['SVM' ,'Random Forest' ,'knn' ,'dt','nb','lr'] 
  
# plotting a bar chart 
plt.bar(left, height, tick_label = tick_label,
               width = 0.5, color = ['purple', 'blue' ,'orange', 'grey','green','red'])


# naming the x-axis 
plt.xlabel('Classifiers') 
# naming the y-axis 
plt.ylabel('Accuracy %') 
# plot title 
plt.title('Accuracy by using different algorithm') 
# function to show the plot 
plt.show()






