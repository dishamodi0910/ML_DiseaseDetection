#import libraries
import numpy as np
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

#loading the data from csv file to a pandas dataframe
parkinsons_data=pd.read_csv('parkinsons.csv') 

#getting number of rows and columns in tha dataset
#parkinsons_data.shape

#distribute of target variable
parkinsons_data['status'].value_counts()

#grouping the data based on the target variable
parkinsons_data.groupby('status').mean()

#separating the features & target
x=parkinsons_data.drop(columns=['name','status'],axis=1)
y=parkinsons_data['status']

#splitting data to training data & test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

#print(x.shape,x_train.shape,x_test.shape)

#data standardlization
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#model tarining
model=svm.SVC()

#training the svm model with tarining data
model.fit(x_train,y_train)

#model ivoluation
#accuracy score
#accuracy score on training data
x_tarin_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_tarin_prediction)

#print('Accuracy score of training data:',training_data_accuracy)

#accuracy score on traing data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(y_test,x_test_prediction)

#print('Accuracy score of testning data:',test_data_accuracy)

#buliding a predictive system
input_data=(202.26600,211.60400,197.07900,0.00180,0.000009,0.00093,0.00107,0.00278,0.00954,0.08500,0.00469,0.00606,0.00719,0.01407,0.00072,32.68400,0.368535,0.742133,-7.695734,0.178540,1.544609,0.056141)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)

prediction=model.predict(std_data)

#0 standes for person does not have disease and 1 stands for person has disease 
print(prediction)
if(prediction[0]==0):
    print("the person does not have parkinsons disease")
else:
    print("the person has parkinsons disease")


