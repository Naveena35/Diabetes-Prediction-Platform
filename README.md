# Diabetes-Prediction-Platform
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
diabetes_datset=pd.read_csv(r"C:\Users\Acer\Downloads\datasets\diabetes.csv")
print(diabetes_datset)
diabetes_datset.head()
diabetes_datset.shape
diabetes_datset.describe()
diabetes_datset['Outcome'].value_counts()
diabetes_datset.groupby('Outcome').mean()
x=diabetes_datset.drop(columns=['Outcome'],axis=1)    #deleting the outcome col and adding separatly in y
y=diabetes_datset['Outcome']
print(x)
print(y)
scaler=StandardScaler()
scaler.fit(x)
standardised=scaler.transform(x)
print(standardised)
x=standardised
y=diabetes_datset['Outcome']
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,stratify=y,random_state=2 ) #0.2=20% of data to be tested,stratify=y is to be same proposion,2-->for splitting in 2
print(x.shape,x_train.shape,x_test.shape)
#training the model
classifier=svm.SVC(kernel='linear')
#training the svc
classifier.fit(x_train,y_train)
#model evaluation
#accuracy score on training data
x_train_prediction=classifier.predict(x_train)
training_accuracy=accuracy_score(x_train_prediction,y_train)
print("accuracy:",training_accuracy)    #ABOVE 75 IS GOOD
#accuracy on testing data like unknown data
x_test_prediction=classifier.predict(x_test)
testing_accuracy=accuracy_score(x_test_prediction,y_test)
print("accuracy:",testing_accuracy)
import warnings
warnings.filterwarnings('ignore')

#making a predictive system
input=(4,110,92,0,0,37.6,0.191,30)
#to convert to the numpy
input_numpy=np.asarray(input)
#reshape for predicting one instance
input_reshape=input_numpy.reshape(1,-1)
#standardize the input data
std_data=scaler.transform(input_reshape)
print(std_data)
prediction=classifier.predict(std_data)
print(prediction)
if(prediction[0]==0):
    print("the person has diabetes")
else:
    print("the person has no diabetes")
