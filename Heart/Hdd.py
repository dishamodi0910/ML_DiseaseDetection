# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

#loading csv file into pandas dataframe
heart_data=pd.read_csv('Heart\heart.csv')



#print(heart_data.head())
#print(heart_data.shape)
#print(heart_data.info())

#checking for null values
#print(heart_data.isnull().sum())

#decribe all information
#print(heart_data.describe)

#print(heart_data['target'].value_counts())

#spliting the feature and targets
x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']

#spliting the data into traing data & test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,stratify=y,random_state=2)

#model for testing
model=LogisticRegression()
#traning the logistic model with traning data
model.fit(x_train,y_train)

#accuracy on traning data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy on training data:',training_data_accuracy)

#accuracy on testing 
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy on test data: ',test_data_accuracy)

#predict that dsease or not
input_data=(66,0,0,178,228,1,1,165,1,1,1,2,3)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print('the person does not have a heart disease')
else:
  print('the person has heart disease')


#generate peackle file
pickle.dump(model,open('Hdd.pkl','wb'))
loaded_model=pickle.load(open,('Hdd.pkl','rb'))
