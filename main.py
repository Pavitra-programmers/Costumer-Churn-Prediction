import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
# for column in df2:  
#         print(f'{column} : {df2[column].unique()}')  
x = df2.drop('Churn',axis='columns')
y = df2['Churn']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=5)


