import pickle
from sklearn.metrics import accuracy_score

pick_out = open(r'C:\Users\dines\OneDrive\Desktop\Github\Costumer Churn\GBTchurnmodel.sav','rb')
data = pickle.load(pick_out)
pick_out.close()

# res = data.predict(x_train)
# print(accuracy_score(y_train,res))

#  0   gender                                    
#  1   SeniorCitizen                        
#  2   Partner                              
#  3   Dependents                           
#  4   tenure                               
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