import sys
import csv
import os
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sexmachine.detector as gender
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
import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
import sys
import pandas as pd
import seaborn as sns
import keras

df=pd.read_csv('../../DSP1/driver_dsp.csv')

#Define Features and Label
features = ['AGE','NUMBER_OF_TRIPS','ACCELERATION','BRAKING','CORNERING','SPEEDING','SEATBELT','DISTRACTION','NUMBER_OF_TICKETS_RECEIVED'] 

X = df[features].values
y = df['SAFETY_SCORE'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 44, shuffle = True)

model = Sequential()

model.add(Dense(9, activation='relu', input_shape=(9,)))
model.add(Dense(9, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=1109, callbacks=[keras.callbacks.EarlyStopping(patience=3)])