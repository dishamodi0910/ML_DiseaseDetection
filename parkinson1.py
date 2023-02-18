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

#distribute of target variable
parkinsons_data['status'].value_counts()

