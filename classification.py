import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import preprocess_data
from feature_extraction import add_features
from sklearn.model_selection import train_test_split

vhdr_file = r"C:\Users\ashut\OneDrive\Desktop\drdo\data\nback1.vhdr"
mkr_file =r"C:\Users\ashut\OneDrive\Desktop\drdo\data\nback1.vmrk"
fname = r"C:\Users\ashut\OneDrive\Desktop\drdo\data\channel_loc.csv"

vhdr_file1 = r"C:\Users\ashut\OneDrive\Desktop\drdo\data\nback2.vhdr"
mkr_file1 =r"C:\Users\ashut\OneDrive\Desktop\drdo\data\nback2.vmrk"

eeg_data = preprocess_data(vhdr_file, mkr_file, fname) 
eeg_data1 = preprocess_data(vhdr_file1, mkr_file1, fname) 

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("feature extraction start")
features_data = add_features(eeg_data)
features_data1 = add_features(eeg_data1)
# data = features_data + features_data1
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("feature extraction done")
data = np.concatenate([features_data, features_data1], axis =0)
df = pd.DataFrame(data)
X=df.iloc[:,1:253]
print("X", X)
y=df[253]
print("Y ", y)
print("X ", X.shape)
print("y ", y.shape)

from sklearn import preprocessing
standard_scaler_object = preprocessing.StandardScaler()
standard_scaler_object.fit(X)
X=standard_scaler_object.transform(X)
X.std(axis=0)

# Splitting the data into train and test 
x_train,x_test,y_train,y_test= train_test_split(X,y, test_size=0.20, random_state=42)

clf = RandomForestClassifier(n_estimators=100, n_jobs=4)
# Train the classifier
clf.fit(x_train, y_train)
y_test_pred=clf.predict(x_test)
acc = clf.score(x_test, y_test)  
print("acc ", acc)
# y_test_pred = model.predict(x_test)
# metrics.accuracy_score(y_test, y_test_pred, normalize=True, sample_weight=None)

# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from preprocessing import preprocess_data
# from feature_extraction import add_features

# Assuming you have already loaded your EEG data into `X` and `y` numpy arrays
# and preprocessed them as necessary

# Create the pipeline
# pipeline = Pipeline([
#     ('cleaning', FunctionTransformer(preprocess_data)),
#     ('feature_extraction', FunctionTransformer(add_features)),
#     ('classification', RandomForestClassifier(n_estimators=100))
# ])

# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Fit the pipeline to the training data
# pipeline.fit(X_train, y_train)

# # Evaluate the pipeline on the test data
# accuracy = pipeline.score(X_test, y_test)