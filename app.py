from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()



app = Flask(__name__)
model = pickle.load(open('GBTchurnmodel.sav', 'rb'))
@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        TotalCharge = int(request.form['TotalCharge'])
        totchar_scaled = scaler.fit_transform([[TotalCharge]])[0][0]
        Monthly_Charges = int(request.form['Monthly_Charges'])
        monchar_scaled = scaler.fit_transform([[Monthly_Charges]])[0][0]
        Tenure = int(request.form['Tenure'])
        tenure_scaled = scaler.fit_transform([[Tenure]])[0][0]
        InternetService = request.form['InternetService']
        if(InternetService == 'DSL'):
            DSL = 1
            Fiber_op= 0
            No = 0
                
        elif(InternetService == 'Fiber Optic'):
            DSL = 0
            Fiber_op= 1
            No = 0
        
        else:
            DSL = 0
            Fiber_op= 0
            No = 1
        Contract = request.form['Contract']
        if(Contract == 'Month-to-Month'):
            mtm = 1
            oney= 0
            twoy = 0
                
        elif(Contract == 'One Year'):
            mtm = 0
            oney= 1
            twoy = 0
        
        else:
            mtm = 0
            oney= 0
            twoy = 1
        PaymentMEthod = request.form['PaymentMEthod']
        if(PaymentMEthod == 'Bank Transfer'):
            BT = 1
            CC= 0
            EC = 0
            C = 0
                
        elif(PaymentMEthod == 'Credit Card'):
            BT = 0
            CC= 1
            EC = 0
            C = 0
        elif(PaymentMEthod == 'Electoric Check'):
             BT = 0
             CC= 0
             EC = 1
             C = 0
        else:
            BT = 0
            CC= 0
            EC = 0
            C = 1
        DeviceProtection = request.form['DevicePro']
        TechSupport = request.form['TechSupport']
        Dependent = request.form['depends']
        Gender = request.form['Gender']
        Partner = request.form['Partner']
        SeniorCitizen = request.form['SeniorCitizen']
        PhoneService = request.form['PhoneService']
        OnlineSecurity = request.form['OnlineSecurity']
        OnlineBackup = request.form['OnlineBackup']
        TVStreaming = request.form['TVStreaming']
        MovieStreaming = request.form['MovieStreaming']
        PaplessBill = request.form['PaplessBill']
        Multiline = request.form['Multiline']
        data1 = [Gender,SeniorCitizen,Partner,Dependent,tenure_scaled,PhoneService,Multiline,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,TVStreaming,MovieStreaming,PaplessBill,monchar_scaled,totchar_scaled,DSL,Fiber_op,No,mtm,oney,twoy,BT,CC,EC,C]
        data2 = np.asarray(data1).reshape(1, -1)
        prediction = model.predict(data2)
        if prediction==1:
             return render_template('index.html',prediction_text="The Customer will leave the Company")
        else:
             return render_template('index.html',prediction_text="The Customer will not leave the Company")
                
if __name__=="__main__":
    app.run(debug=True)