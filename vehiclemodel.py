import sys
import csv
import os
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sexmachine.detector
from sklearn.impute import SimpleImputer
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# %matplotlib inline 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
import sys
import pandas as pd
import seaborn as sns



df=pd.read_csv('../../DSP1/vehicle_dsp.csv')
def clean_cap(x):
  if x == 'STD':
    return 1
  if x == 'LARGE':
    return 2
  if x == 'XL LARGE':
    return 3
df['CAPACITY'] = df['CAPACITY'].apply(clean_cap)

def clean_battery(x):
  if x == 'AVERAGE':
    return 1
  if x == 'GOOD':
    return 2
df['BATTERY_HEALTH'] = df['BATTERY_HEALTH'].apply(clean_battery)

for i in range(len(df)):
  if df['CAPACITY'][i] == 'STD':
    df['CAPACITY'][i] = int(1)
  elif df['CAPACITY'][i] == 'LARGE':
    df['CAPACITY'][i] = int(2)
  elif df['CAPACITY'][i] == 'XL LARGE':
    df['CAPACITY'][i] = int(3)

for i in range(len(df)):
  if df['BATTERY_HEALTH'][i] == 'AVERAGE':
    df['BATTERY_HEALTH'][i] = int(1)
  elif df['BATTERY_HEALTH'][i] == 'GOOD':
    df['BATTERY_HEALTH'][i] = int(2)

# df.drop([250,250],axis=0,inplace=True)

X = df.drop(columns=['ID','LICENSE_PLATE','REGISTERED_STATE','MAKE','MODEL','YEAR','TOTAL_MILES_DONE','SAFETY_SCORE','DASH_CAM_IP','LAST_SERVICE_DATE','LAST_SERVICE_MILES','NEXT_SERVICE_DATE','NEXT_SERVICE_MILES']).values
Y = df[['SAFETY_SCORE']].values

from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

#Define Features and Label
features = ['CAPACITY','BATTERY_HEALTH','BATTERY_VOLTAGE','TYRE_PRESSURE','FUEL_LEVEL','OIL_LEVEL'] 

X=df[features].values
y=df['SAFETY_SCORE'].values

#Train Test Split
# X_train=np.asarray(X_train).astype(np.int)

# y_train=np.asarray(y_train).astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 44, shuffle = True)

#Model
model = Sequential()
model.add(Dense(6, activation='relu', kernel_initializer='random_normal', input_dim = 6))
model.add(Dense(1, activation = 'relu', kernel_initializer='random_normal'))

#Compiling the neural network
model.compile(optimizer = Adam(learning_rate=0.1) ,loss='mean_squared_logarithmic_error', metrics =['mse'])

model.fit(X_train,y_train, batch_size=256, epochs=1109, verbose=0)