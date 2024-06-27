import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import pickle

df = pd.read_csv("Churn-Data.csv") #reading the CSV file

df.drop('cID',axis='columns',inplace=True) #removing the Cid Column
df1 = df[df.TotalCharges!=' '] #tranfering not null value to new datframe
df1.TotalCharges = pd.to_numeric(df1.TotalCharges) #Converting object datatype to floate datatype
pd.set_option('future.no_silent_downcasting', True)

#Replacing No internet service and No phone service into No
for column in df1:  
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

#splitting the dataset
x = df2.drop('Churn',axis='columns')
y = df2['Churn']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=5)


#prediction through gradient boosting classifier
gbt = GradientBoostingClassifier()
gbt.fit(x,y)
yp4 = gbt.predict(x_test)
#accuracy and other score
print("Gradient Boosting model",accuracy_score(y_test,yp4),precision_score(y_test,yp4),recall_score(y_test,yp4),f1_score(y_test,yp4))
print(yp4[:5],y_test[:5])

pick = open('GBTchurnmodel.sav','wb')
pickle.dump(gbt,pick)
pick.close()


