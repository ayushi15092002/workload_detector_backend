import numpy as np
import pandas as pd
import mne
from features.coeff_of_variation import coeff_var
from features.first_diff_max import first_diff_max
from features.first_diff_mean import first_diff_mean
from features.fractal_dimension import fractal_dimension
from features.hjorth import hjorth
from features.kurtosis import kurtosis
from features.max_power import maxPwelch
from features.mean_vertex_to_vertex_slope import slope_mean
from features.sec_diff_max import secDiffMax
from features.sec_diff_mean import secDiffMean
from features.skewness import skewness
from features.variance_vertex_to_vertex_slope import slope_var
from features.wavelet_features import wavelet_features

# Arrays of column names(features that we are extracting)
names = ['Fractal_Dimension','Coeffiecient_of_Variation',
         'Mean_of_Vertex_to_Vertex_Slope','Variance_of_Vertex_to_Vertex_Slope',
         'Hjorth_Activity','Hjorth_Mobility','Hjorth_Complexity',
         'Kurtosis','2nd_Difference_Mean','2nd_Difference_Max',
         'Skewness','1st_Difference_Mean','1st_Difference_Max',
        #  'FFT_Delta_MaxPower','FFT_Theta_MaxPower','FFT_Alpha_MaxPower','FFT_Beta_MaxPower','Delta/Theta','Delta/Alpha','Theta/Alpha','(Delta+Theta)/Alpha',
         '1_Wavelet_Approximate_Mean','1_Wavelet_Approximate_Std_Deviation','1_Wavelet_Approximate_Energy','1_Wavelet_Detailed_Mean','1_Wavelet_Detailed_Std_Deviation','1_Wavelet_Detailed_Energy','1_Wavelet_Approximate_Entropy','1_Wavelet_Detailed_Entropy',
         '2_Wavelet_Approximate_Mean','2_Wavelet_Approximate_Std_Deviation','2_Wavelet_Approximate_Energy','2_Wavelet_Detailed_Mean','2_Wavelet_Detailed_Std _Deviation','2_Wavelet_Detailed_Energy','2_Wavelet_Approximate_Entropy','2_Wavelet_Detailed_Entropy',
         '3_Wavelet_Approximate_Mean','3_Wavelet_Approximate_Std_Deviation','3_Wavelet_Approximate_Energy','3_Wavelet_Detailed_Mean','3_Wavelet_Detailed_Std _Deviation','3_Wavelet_Detailed_Energy','3_Wavelet_Approximate_Entropy','3_Wavelet_Detailed_Entropy',
         '4_Wavelet_Approximate_Mean','4_Wavelet_Approximate_Std_Deviation','4_Wavelet_Approximate_Energy','4_Wavelet_Detailed_Mean','4_Wavelet_Detailed_Std _Deviation','4_Wavelet_Detailed_Energy','4_Wavelet_Approximate_Entropy','4_Wavelet_Detailed_Entropy',
         '5_Wavelet_Approximate_Mean','5_Wavelet_Approximate_Std_Deviation','5_Wavelet_Approximate_Energy','5_Wavelet_Detailed_Mean','5_Wavelet_Detailed_Std _Deviation','5_Wavelet_Detailed_Energy','5_Wavelet_Approximate_Entropy','5_Wavelet_Detailed_Entropy',
         '6_Wavelet_Approximate_Mean','6_Wavelet_Approximate_Std_Deviation','6_Wavelet_Approximate_Energy','6_Wavelet_Detailed_Mean','6_Wavelet_Detailed_Std _Deviation','6_Wavelet_Detailed_Energy','6_Wavelet_Approximate_Entropy','6_Wavelet_Detailed_Entropy',
         '7_Wavelet_Approximate_Mean','7_Wavelet_Approximate_Std_Deviation','7_Wavelet_Approximate_Energy','7_Wavelet_Detailed_Mean','7_Wavelet_Detailed_Std _Deviation','7_Wavelet_Detailed_Energy','7_Wavelet_Approximate_Entropy','7_Wavelet_Detailed_Entropy',
         '8_Wavelet_Approximate_Mean','8_Wavelet_Approximate_Std_Deviation','8_Wavelet_Approximate_Energy','8_Wavelet_Detailed_Mean','8_Wavelet_Detailed_Std _Deviation','8_Wavelet_Detailed_Energy','8_Wavelet_Approximate_Entropy','8_Wavelet_Detailed_Entropy',
         '9_Wavelet_Approximate_Mean','9_Wavelet_Approximate_Std_Deviation','9_Wavelet_Approximate_Energy','9_Wavelet_Detailed_Mean','9_Wavelet_Detailed_Std _Deviation','9_Wavelet_Detailed_Energy','9_Wavelet_Approximate_Entropy','9_Wavelet_Detailed_Entropy',
         '10_Wavelet_Approximate_Mean','10_Wavelet_Approximate_Std_Deviation','10_Wavelet_Approximate_Energy','10_Wavelet_Detailed_Mean','10_Wavelet_Detailed_Std _Deviation','10_Wavelet_Detailed_Energy','10_Wavelet_Approximate_Entropy','10_Wavelet_Detailed_Entropy',
         '11_Wavelet_Approximate_Mean','11_Wavelet_Approximate_Std_Deviation','11_Wavelet_Approximate_Energy','11_Wavelet_Detailed_Mean','11_Wavelet_Detailed_Std _Deviation','11_Wavelet_Detailed_Energy','11_Wavelet_Approximate_Entropy','11_Wavelet_Detailed_Entropy',
         '12_Wavelet_Approximate_Mean','12_Wavelet_Approximate_Std_Deviation','12_Wavelet_Approximate_Energy','12_Wavelet_Detailed_Mean','12_Wavelet_Detailed_Std _Deviation','12_Wavelet_Detailed_Energy','12_Wavelet_Approximate_Entropy','12_Wavelet_Detailed_Entropy',
         '13_Wavelet_Approximate_Mean','13_Wavelet_Approximate_Std_Deviation','13_Wavelet_Approximate_Energy','13_Wavelet_Detailed_Mean','13_Wavelet_Detailed_Std _Deviation','13_Wavelet_Detailed_Energy','13_Wavelet_Approximate_Entropy','13_Wavelet_Detailed_Entropy',
         '14_Wavelet_Approximate_Mean','14_Wavelet_Approximate_Std_Deviation','14_Wavelet_Approximate_Energy','14_Wavelet_Detailed_Mean','14_Wavelet_Detailed_Std _Deviation','14_Wavelet_Detailed_Energy','14_Wavelet_Approximate_Entropy','14_Wavelet_Detailed_Entropy',
         '15_Wavelet_Approximate_Mean','15_Wavelet_Approximate_Std_Deviation','15_Wavelet_Approximate_Energy','15_Wavelet_Detailed_Mean','15_Wavelet_Detailed_Std _Deviation','15_Wavelet_Detailed_Energy','15_Wavelet_Approximate_Entropy','15_Wavelet_Detailed_Entropy',
         '16_Wavelet_Approximate_Mean','16_Wavelet_Approximate_Std_Deviation','16_Wavelet_Approximate_Energy','16_Wavelet_Detailed_Mean','16_Wavelet_Detailed_Std _Deviation','16_Wavelet_Detailed_Energy','16_Wavelet_Approximate_Entropy','16_Wavelet_Detailed_Entropy',
         '17_Wavelet_Approximate_Mean','17_Wavelet_Approximate_Std_Deviation','17_Wavelet_Approximate_Energy','17_Wavelet_Detailed_Mean','17_Wavelet_Detailed_Std _Deviation','17_Wavelet_Detailed_Energy','17_Wavelet_Approximate_Entropy','17_Wavelet_Detailed_Entropy',
         '18_Wavelet_Approximate_Mean','18_Wavelet_Approximate_Std_Deviation','18_Wavelet_Approximate_Energy','18_Wavelet_Detailed_Mean','18_Wavelet_Detailed_Std _Deviation','18_Wavelet_Detailed_Energy','18_Wavelet_Approximate_Entropy','18_Wavelet_Detailed_Entropy',
         '19_Wavelet_Approximate_Mean','19_Wavelet_Approximate_Std_Deviation','19_Wavelet_Approximate_Energy','19_Wavelet_Detailed_Mean','19_Wavelet_Detailed_Std _Deviation','19_Wavelet_Detailed_Energy','19_Wavelet_Approximate_Entropy','19_Wavelet_Detailed_Entropy',
         '20_Wavelet_Approximate_Mean','20_Wavelet_Approximate_Std_Deviation','20_Wavelet_Approximate_Energy','20_Wavelet_Detailed_Mean','20_Wavelet_Detailed_Std _Deviation','20_Wavelet_Detailed_Energy','20_Wavelet_Approximate_Entropy','20_Wavelet_Detailed_Entropy',
         '21_Wavelet_Approximate_Mean','21_Wavelet_Approximate_Std_Deviation','21_Wavelet_Approximate_Energy','21_Wavelet_Detailed_Mean','21_Wavelet_Detailed_Std _Deviation','21_Wavelet_Detailed_Energy','21_Wavelet_Approximate_Entropy','21_Wavelet_Detailed_Entropy',
         '22_Wavelet_Approximate_Mean','22_Wavelet_Approximate_Std_Deviation','22_Wavelet_Approximate_Energy','22_Wavelet_Detailed_Mean','22_Wavelet_Detailed_Std _Deviation','22_Wavelet_Detailed_Energy','22_Wavelet_Approximate_Entropy','22_Wavelet_Detailed_Entropy',
         '23_Wavelet_Approximate_Mean','23_Wavelet_Approximate_Std_Deviation','23_Wavelet_Approximate_Energy','23_Wavelet_Detailed_Mean','23_Wavelet_Detailed_Std _Deviation','23_Wavelet_Detailed_Energy','23_Wavelet_Approximate_Entropy','23_Wavelet_Detailed_Entropy',
         '24_Wavelet_Approximate_Mean','24_Wavelet_Approximate_Std_Deviation','24_Wavelet_Approximate_Energy','24_Wavelet_Detailed_Mean','24_Wavelet_Detailed_Std _Deviation','24_Wavelet_Detailed_Energy','24_Wavelet_Approximate_Entropy','24_Wavelet_Detailed_Entropy',
         '25_Wavelet_Approximate_Mean','25_Wavelet_Approximate_Std_Deviation','25_Wavelet_Approximate_Energy','25_Wavelet_Detailed_Mean','25_Wavelet_Detailed_Std _Deviation','25_Wavelet_Detailed_Energy','25_Wavelet_Approximate_Entropy','25_Wavelet_Detailed_Entropy',
         '26_Wavelet_Approximate_Mean','26_Wavelet_Approximate_Std_Deviation','26_Wavelet_Approximate_Energy','26_Wavelet_Detailed_Mean','26_Wavelet_Detailed_Std _Deviation','26_Wavelet_Detailed_Energy','26_Wavelet_Approximate_Entropy','26_Wavelet_Detailed_Entropy',
         '27_Wavelet_Approximate_Mean','27_Wavelet_Approximate_Std_Deviation','27_Wavelet_Approximate_Energy','27_Wavelet_Detailed_Mean','27_Wavelet_Detailed_Std _Deviation','27_Wavelet_Detailed_Energy','27_Wavelet_Approximate_Entropy','27_Wavelet_Detailed_Entropy',
         '28_Wavelet_Approximate_Mean','28_Wavelet_Approximate_Std_Deviation','28_Wavelet_Approximate_Energy','28_Wavelet_Detailed_Mean','28_Wavelet_Detailed_Std _Deviation','28_Wavelet_Detailed_Energy','28_Wavelet_Approximate_Entropy','28_Wavelet_Detailed_Entropy',
         '29_Wavelet_Approximate_Mean','29_Wavelet_Approximate_Std_Deviation','29_Wavelet_Approximate_Energy','29_Wavelet_Detailed_Mean','29_Wavelet_Detailed_Std _Deviation','29_Wavelet_Detailed_Energy','29_Wavelet_Approximate_Entropy','29_Wavelet_Detailed_Entropy',
         '30_Wavelet_Approximate_Mean','30_Wavelet_Approximate_Std_Deviation','30_Wavelet_Approximate_Energy','30_Wavelet_Detailed_Mean','30_Wavelet_Detailed_Std _Deviation','30_Wavelet_Detailed_Energy','30_Wavelet_Approximate_Entropy','30_Wavelet_Detailed_Entropy',
        #  'Shannon_Entropy', 'Hurst_Exponent', 'Permutation_Entropy'
         'y']
        #  ,'detrended fluctuation analysis (DFA)']



# list to store files
# res = []
# dir_path = "/content/drive/My Drive/mental workload/epochs_data"
# Iterate directory
# for path in os.listdir(dir_path):
#     # check if current path is a file
#     if os.path.isfile(os.path.join(dir_path, path)):
#         res.append(path)

# for f in res:
#   add_features(dir_path + "/"+f)

# # add_features("/content/drive/My Drive/mental workload/epochs_data/VP002_1.fif")
# # print(len(x))
# add_features("/content/drive/My Drive/mental workload/epochs_data/VP002_2.fif")
# print(len(x))
# add_features("/content/drive/My Drive/mental workload/epochs_data/VP002_3.fif")
# print(len(x))

# This function takes the path of the file as a input and extract the featurs from the file 

def add_features(epochs):
  # epochs = mne.read_epochs(fname=path,verbose=False)
  nchans = epochs.info['nchan']
  events = epochs.events
  y_values = events[:,-1]
  dat = []
  x=[]
  # x.append(names)
  j=0
  
  for i in epochs:
    dat.append(np.array(i))

  dat = np.array(dat)

  tr = dat.shape[1]

  for i in dat:
    print("i ", i)
    features=[]
    # Fractal Dimension
    features.append(fractal_dimension(i,tr))

    #Coeffeicient of Variation
    features.append(coeff_var(i,tr))

    # Mean of Vertex to Vertex Slope
    features.append(slope_mean(i,tr))

    # Variance of Vertex to Vertex Slope
    
    features.append(slope_var(i,tr))

    #Hjorth Parameters
    feature_list = hjorth(i,tr)
    for feat in feature_list:
        features.append(feat)

    #Kurtosis
    features.append(kurtosis(i,tr))

    #Second Difference Mean
    features.append(secDiffMean(i,tr))

    #Second Difference Max
    features.append(secDiffMax(i,tr))

    #Skewness
    features.append(skewness(i,tr))

    #First Difference Mean
    features.append(first_diff_mean(i,tr))

    #First Difference Max
    features.append(first_diff_max(i,tr))

    # wavlent features
    feature_list = wavelet_features(i,nchans)
    for feat in feature_list:
        features.append(feat)
    
    features.append(y_values[j])
    j = j+1
    x.append(features)

  pd.DataFrame(x)
  return x





