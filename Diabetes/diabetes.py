#import the libraries

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
from sklearn import svm

#load the data
dataset = pd.read_csv('Diabetes\diabetes.csv')  
# print(dataset.head())
# print(dataset.isnull().sum()

#split the dependent data
x=dataset.drop(columns='Outcome',axis=1)  

#split the independent data
y=dataset['Outcome'] 

# split the train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

model = svm.SVC(kernel='linear')

#fit the model
model.fit(x_train,y_train)

# print(model.score(x_test,y_test))

input = (6,104,74,18,156,29.9,0.722,41)
input1 = np.asarray(input)
input2 = input1.reshape(1,-1)
ans=model.predict(input2)

if(ans[0]==1):
   print("person have diabetes.")
else:
   print("person does not have diabetes.")
