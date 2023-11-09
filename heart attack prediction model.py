import tkinter
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn import preprocessing
# %matplotlib inline

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier

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


#print("Accuracy of SVM Algorithm: {:.2f}%".format(svclassifier.score(X_test,y_test)*100))

svm = svclassifier.score(X_test, y_test)*100


  # train model
scaler.fit_transform(X_train,y_train)
fit_accuracy = svclassifier.score(X_train, y_train)
test_accuracy = svclassifier.score(X_test, y_test)

def takeInput():
    inputValues = []


    age1 = ((int(age.get()) - 29)  / (77-29 ))
    print(age1)
    trestbps1 = ((int(rbp.get()) - 94)/(200-94))
    chol1 = ((int (serumChol.get()) - 126)/(564-126))
    thalach1 = ((int(thalach.get()) - 71)/(202-71))
    oldpeak1 = (int(oldpeak.get())/ (3))

    inputValues.append(age1)
    inputValues.append(sex.get())
    inputValues.append(chestPain.get())
    inputValues.append(trestbps1)
    inputValues.append(chol1)
    inputValues.append(FBS.get())
    inputValues.append(ECG.get())
    inputValues.append(thalach1)
    inputValues.append(trestbps1)
    inputValues.append(oldpeak1)
    inputValues.append(slope.get())
    inputValues.append(ca.get())
    inputValues.append(thal.get()) 

    print(inputValues)


    print("\n") 
    
  
    final_Result = svclassifier.predict(X_test)
    print(final_Result)

    substituteWindow = tkinter.Tk()
    substituteWindow.geometry('640x480-8-200')
    substituteWindow.title("RESULT PREDICTION")

    substituteWindow.columnconfigure(0, weight=2)
    substituteWindow.columnconfigure(1, weight=1)
    substituteWindow.columnconfigure(2, weight=2)
    substituteWindow.columnconfigure(3, weight=2)
    substituteWindow.rowconfigure(0, weight=1)
    substituteWindow.rowconfigure(1, weight=10)
    substituteWindow.rowconfigure(2, weight=10)
    substituteWindow.rowconfigure(3, weight=1)
    substituteWindow.rowconfigure(4, weight=1)
    substituteWindow.rowconfigure(5, weight=1)

    if final_Result[0] == 1:
        label1 = tkinter.Label(substituteWindow, text="HEART ATTACK DETECTED", font=('Impact', -35), fg='#0080ff')
        label1.grid(row=0, column=1, columnspan=6)
        label2 = tkinter.Label(substituteWindow, text="PLEASE VISIT NEAREST CARDIOLOGIST AT THE EARLIEST", font=('Impact', -20), fg='red')
        label2.grid(row=1, column=1, columnspan=6)
        
    else: 
        label1 = tkinter.Label(substituteWindow, text="NO DETECTION OF HEART ATTACK", font=('Impact', -35) )
        label1.grid(row=2, column=1, columnspan=6)
        label2 = tkinter.Label(substituteWindow, text="Do not forget to exercise daily. ", font=('Impact', -20), fg='green')
        label2.grid(row=3, column=1, columnspan=6)      

    substituteWindow.mainloop()

mainWindow = tkinter.Tk()
mainWindow.geometry('640x480-8-200')
mainWindow['padx']=20
mainWindow.title("HEART ATTACK PREDICTION")

mainWindow.columnconfigure(0, weight=2)
mainWindow.columnconfigure(1, weight=1)
mainWindow.columnconfigure(2, weight=2)
mainWindow.columnconfigure(3, weight=2)
mainWindow.rowconfigure(0, weight=0)
mainWindow.rowconfigure(1, weight=0)
mainWindow.rowconfigure(2, weight=1)
mainWindow.rowconfigure(3, weight=1)
mainWindow.rowconfigure(4, weight=1)
mainWindow.rowconfigure(5, weight=1)
mainWindow.rowconfigure(6, weight=1)
mainWindow.rowconfigure(7, weight=1)
mainWindow.rowconfigure(8, weight=10)


label1 = tkinter.Label(mainWindow, text="HEART ATTACK PREDICTION MODEL", font=('Impact', -35), bg='#ff8000')
label1.grid(row=0, column=0, columnspan=6)

label2 = tkinter.Label(mainWindow, text="Enter the details carefully", font=('Impact', -20) , fg='white', bg='#ff00bf' )
label2.grid(row=1, column=0, columnspan=6)


#frame for the feature inputs
ageFrame = tkinter.LabelFrame(mainWindow, text="Age(yrs)")
ageFrame.grid(row=2, column=0)
ageFrame.config(font=("Courier", -15))
age= tkinter.Entry(ageFrame)
age.grid(row=2, column=2, sticky='nw')

sexFrame = tkinter.LabelFrame(mainWindow, text="Sex")
sexFrame.grid(row=2, column=1)
sexFrame.config(font=("Courier", -15))
sex= tkinter.Entry(sexFrame)
sex.grid(row=2, column=2, sticky='nw')

chestPainFrame = tkinter.LabelFrame(mainWindow, text="CP (1-4)")
chestPainFrame.grid(row=2, column=2)
chestPainFrame.config(font=("Courier", -15))
chestPain= tkinter.Entry(chestPainFrame)
chestPain.grid(row=2, column=2, sticky='nw')


rbpFrame = tkinter.LabelFrame(mainWindow, text="RBP (94-200)")
rbpFrame.grid(row=3, column=0)
rbpFrame.config(font=("Courier", -15))
rbp= tkinter.Entry(rbpFrame)
rbp.grid(row=2, column=2, sticky='nw')

serumCholFrame = tkinter.LabelFrame(mainWindow, text="Serum Chol(126-564)")
serumCholFrame.grid(row=3, column=1)
serumCholFrame.config(font=("Courier", -15))
serumChol = tkinter.Entry(serumCholFrame)
serumChol.grid(row=2, column=2, sticky='n')

FBSFrame = tkinter.LabelFrame(mainWindow, text="Fasting Bs(0,1)")
FBSFrame.grid(row=3, column=2)
FBSFrame.config(font=("Courier", -15))
FBS= tkinter.Entry(FBSFrame)
FBS.grid(row=2, column=2, sticky='nw')

ECGFrame = tkinter.LabelFrame(mainWindow, text="ECG (0,1,2)")
ECGFrame.grid(row=4, column=0)
ECGFrame.config(font=("Courier", -15))
ECG = tkinter.Entry(ECGFrame)
ECG.grid(row=2, column=2, sticky='nw')


thalachFrame = tkinter.LabelFrame(mainWindow, text="thalach(71-202)")
thalachFrame.grid(row=4, column=1)
thalachFrame.config(font=("Courier", -15))
thalach = tkinter.Entry(thalachFrame)
thalach.grid(row=2, column=2, sticky='nw')

exangFrame = tkinter.LabelFrame(mainWindow, text="exAngina(0/1)")
exangFrame.grid(row=4, column=2)
exangFrame.config(font=("Courier", -15))
exang = tkinter.Entry(exangFrame)
exang.grid(row=2, column=2, sticky='nw')


oldpeakFrame = tkinter.LabelFrame(mainWindow, text="Old Peak(1-3)")
oldpeakFrame.grid(row=5, column=0)
oldpeakFrame.config(font=("Courier", -15))
oldpeak = tkinter.Entry(oldpeakFrame)
oldpeak.grid(row=2, column=2, sticky='nw')

slopeFrame = tkinter.LabelFrame(mainWindow, text="Slope(1,2,3)")
slopeFrame.grid(row=5, column=1)
slopeFrame.config(font=("Courier", -15))
slope = tkinter.Entry(slopeFrame)
slope.grid(row=2, column=2, sticky='nw')

caFrame = tkinter.LabelFrame(mainWindow, text=" C. A (0-3)")
caFrame.grid(row=5, column=2)
caFrame.config(font=("Courier", -15))
ca = tkinter.Entry(caFrame)
ca.grid(row=2, column=2, sticky='nw')


thalFrame = tkinter.LabelFrame(mainWindow, text=" THAL(3,6,7)")
thalFrame.grid(row=6, column=1)
thalFrame.config(font=("Courier", -15))
thal = tkinter.Entry(thalFrame)
thal.grid(row=2, column=2, sticky='nw')


analyseButton = tkinter.Button(mainWindow, text="..................ANALYZE/ PREDICT.....................", font=('Impact', -15), bg = 'red', command=takeInput)
analyseButton.grid(row=8, column=0, columnspan=10)



mainWindow.mainloop()



