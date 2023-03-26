import pywt
import numpy as np

def wavelet_features(epoch,channels):
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy =[]
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = [] 
    Entropy_A = []
    wfeatures = []
    for i in range(channels):
        cA,cD=pywt.dwt(epoch[i,:],'coif1')
        cA_values.append(cA)
        cD_values.append(cD)		#calculating the coefficients of wavelet transform.
    for x in range(channels):   
        cA_mean.append(np.mean(cA_values[x]))
        wfeatures.append(np.mean(cA_values[x]))
        
        cA_std.append(abs(np.std(cA_values[x])))
        wfeatures.append(abs(np.std(cA_values[x])))
        
        cA_Energy.append(abs(np.sum(np.square(cA_values[x]))))
        wfeatures.append(abs(np.sum(np.square(cA_values[x]))))
        
        cD_mean.append(np.mean(cD_values[x]))		# mean and standard deviation values of coefficents of each channel is stored .
        wfeatures.append(np.mean(cD_values[x]))

        cD_std.append(abs(np.std(cD_values[x])))
        wfeatures.append(abs(np.std(cD_values[x])))
        
        cD_Energy.append(abs(np.sum(np.square(cD_values[x]))))
        wfeatures.append(abs(np.sum(np.square(cD_values[x]))))
        
        Entropy_D.append(abs(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x])))))
        wfeatures.append(abs(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x])))))
        
        Entropy_A.append(abs(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x]))))) 
        wfeatures.append(abs(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x])))))
    return wfeatures

