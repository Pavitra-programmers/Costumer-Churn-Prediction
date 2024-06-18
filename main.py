import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("Churn-Data.csv") #reading the CSV file
# print(df.sample(5))
df.drop('cID',axis='columns',inplace=True) #removing the Cid Column
df1 = df[df.TotalCharges!=' '] #tranfering not null value to new datframe
df1.TotalCharges = pd.to_numeric(df1.TotalCharges) #Converting object datatype to floate datatype
pd.set_option('future.no_silent_downcasting', True)

for column in df1:  #Replacing No internet service and No phone service into No
    df1[column] = df1[column].replace('No internet service', 'No')
    df1[column] = df1[column].replace('No phone service', 'No')
for column in [ 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'TV_Streaming', 'Movie_Streaming','PaperlessBilling', 'Churn',]:
    df1[column] = df1[column].map({'Yes':1,'No':0})
df1['gender'] = df1['gender'].map({'Female':1,'Male':0})
#one hot encoding method
df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','Method_Payment'], dtype=int)

cols_to_scale = ['tenure','Charges_Month','TotalCharges'] #minimizing the values

scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
# for column in df2:  
#         print(f'{column} : {df2[column].unique()}') 
#splittinf the df
x = df2.drop('Churn',axis='columns')
y = df2['Churn']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=5)
#making the tf model
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=100)
yp = model.predict(x_test)
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
#giving result into 1 and 0
res = []
for ele in yp:
    if ele > 0.5:
        res.append(1)
    else:
        res.append(0)
#prediction through SVC method
from sklearn import svm
svm = svm.SVC()
svm.fit(x_train,y_train)
yp1 = svm.predict(x_test)
#prediction through decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
yp2 = dt.predict(x_test)
#prediction through random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
yp3 = rf.predict(x_test)
#prediction through gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier()
gbt.fit(x_train,y_train)
yp4 = gbt.predict(x_test)
#accuracy and other score
print("Tensorflow model",accuracy_score(y_test,res),precision_score(y_test,res),recall_score(y_test,res),f1_score(y_test,res))
print("SVC model",accuracy_score(y_test,yp1),precision_score(y_test,yp1),recall_score(y_test,yp1),f1_score(y_test,yp1))
print("Decision tree model",accuracy_score(y_test,yp2),precision_score(y_test,yp2),recall_score(y_test,yp2),f1_score(y_test,yp2))
print("Random Forest model",accuracy_score(y_test,yp3),precision_score(y_test,yp3),recall_score(y_test,yp3),f1_score(y_test,yp3))
print("Gradient Boosting model",accuracy_score(y_test,yp4),precision_score(y_test,yp4),recall_score(y_test,yp4),f1_score(y_test,yp4))
# print(res[:5])
# print(y_test[:5])

# print(x_test.dtypes)
