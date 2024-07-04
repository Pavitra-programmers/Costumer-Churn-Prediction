import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the model
pick_out = open(r'C:\Users\dines\OneDrive\Desktop\Github\Costumer Churn\GBTchurnmodel.sav','rb')
Model = pickle.load(pick_out)
pick_out.close()

# Initialize the scaler
scaler = MinMaxScaler()

# Transform the individual values
tenure = 21
tenure_scaled = scaler.fit_transform([[tenure]])[0][0]

totchar = 1336.8
totchar_scaled = scaler.fit_transform([[totchar]])[0][0]

monchar = 64.85
monchar_scaled = scaler.fit_transform([[monchar]])[0][0]

# Create the data array
data1 = [0, 0, 0, 1, tenure_scaled, 1, 0, 1, 0, 1, 0, 0, 1, 0, monchar_scaled, totchar_scaled, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
data2 = np.asarray(data1).reshape(1, -1)

# Make a prediction
prediction = Model.predict(data2)
#display the results
if prediction[0] == 1:
    print('The costumer will leave the company.')
else:
    print('The costumer will not leave the company.')