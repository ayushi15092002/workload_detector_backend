import pickle
import joblib
from flask import Flask, render_template, request
from sklearn import preprocessing
from preprocessing import preprocess_data
from feature_extraction import add_features
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Create an object f the flask class
app = Flask(__name__, template_folder='template')
model  = joblib.load(open('rfc_model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # get files from request
    vhdr_file = request.files['vhdr']
    mkr_file = request.files['vmrk']
    eeg_file = request.files['eeg']
    # save files to temporary location
    # nback1 = vhdr_file
    vhdr_path = r"F:/workload_detector_backend/temp_data/nback1.vhdr"
    vhdr_file.save(vhdr_path)

    # nback1 = mkr_file
    vmrk_path = r"F:/workload_detector_backend/temp_data/nback1.vmrk"
    mkr_file.save(vmrk_path)

    # nback1 = eeg_file
    eeg_path = r"F:/workload_detector_backend/temp_data/nback1.eeg"
    eeg_file.save(eeg_path)

    # vhdr_file = r"data\nback1.vhdr"
    # mkr_file =r"data\nback1.vmrk"
    fname = r"data\channel_loc.csv"


    eeg_data = preprocess_data(vhdr_path, vmrk_path, fname) 
    features_data = add_features(eeg_data)
    df = pd.DataFrame(features_data)
    X=df.iloc[:,1:253]
    y=df[253]
    print("X ", X.shape)
    print("y ", y.shape)

    standard_scaler_object = preprocessing.StandardScaler()
    standard_scaler_object.fit(X)
    X=standard_scaler_object.transform(X)
    X.std(axis=0)

    prediction = model.predict((X))
    l = np.array(prediction)
    int_array = np.round(l).astype(np.int64)
    result = np.bincount(int_array).argmax()
    print("prediction ",prediction)
    print("max ", result )

    return str(result)

if __name__ == '__main__':
    app.run(debug=True)



