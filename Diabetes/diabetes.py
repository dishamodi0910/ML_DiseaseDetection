#import the libraries

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler
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

# standardlize the data
sc = StandardScaler()
sc.fit(x)
s_data = sc.transform(x)
# print(s_data)

x = s_data
y=dataset['Outcome']

# split the train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

model = svm.SVC(kernel='linear')

# fit the data into the model
model.fit(x_train,y_train)

pickle.dump(model,open('diabetes.pkl','wb'))
load_model = pickle.load(open('diabetes.pkl','rb'))

# print(model.score(x_test,y_test))

# predict the output 
input = (4,110,66,0,0,31.9,0.471,29)
input1 = np.asarray(input)
input2 = input1.reshape(1,-1)
std_data = sc.transform(input2)
ans=model.predict(std_data)

if(ans[0]==1):
   print("person have diabetes.")
else:
   print("person does not have diabetes.")
