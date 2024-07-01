import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the model
pick_out = open(r'C:\Users\dines\OneDrive\Desktop\Github\Costumer Churn\GBTchurnmodel.sav','rb')
data = pickle.load(pick_out)
pick_out.close()

# Initialize the scaler
scaler = MinMaxScaler()

# Example sample data for fitting the scaler
sample_data = np.array([[1], [5], [10], [20], [30], [50]])

# Fit the scaler on the sample data
scaler.fit(sample_data)

# Transform the individual values
tenure = 21
tenure_scaled = scaler.transform([[tenure]])[0][0]

totchar = 1336.8
totchar_scaled = scaler.transform([[totchar]])[0][0]

monchar = 64.85
monchar_scaled = scaler.transform([[monchar]])[0][0]

# Create the data array
data1 = [0, 0, 0, 1, tenure_scaled, 1, 0, 1, 0, 1, 0, 0, 1, 0, monchar_scaled, totchar_scaled, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
data2 = np.asarray(data1).reshape(1, -1)

# Make a prediction
prediction = data.predict(data2)
print(prediction)

# res = data.predict(x_train)
# print(accuracy_score(y_train,res))

#  0   gender  1                                  
#  1   SeniorCitizen                        
#  2   Partner                              
#  3   Dependents                           
#  4   tenure   1                      
#  5   PhoneService                         
#  6   MultipleLines                        
#  7   OnlineSecurity                       
#  8   OnlineBackup                         
#  9   DeviceProtection                     
#  10  TechSupport                          
#  11  TV_Streaming                         
#  12  Movie_Streaming                       
#  13  PaperlessBilling                      
#  14  Charges_Month                         
#  15  TotalCharges                          
#  16  InternetService_DSL                   
#  17  InternetService_Fiber optic           
#  18  InternetService_No                    
#  19  Contract_Month-to-month               
#  20  Contract_One year                     
#  21  Contract_Two year                     
#  22  Method_Payment_Bank transfer (automatic)
#  23  Method_Payment_Credit card (automatic)  
#  24  Method_Payment_Electronic check         
#  25  Method_Payment_Mailed check       