from scipy import signal
import numpy as np

def maxPwelch(data_win,Fs,tr):
    BandF = [0.1, 3, 7, 12, 30]
    PMax = np.zeros([tr,(len(BandF)-1)]);
    
    for j in range(14):
        f,Psd = signal.welch(data_win[j,:], Fs)
        
        for i in range(len(BandF)-1):
            fr = np.where((f>BandF[i]) & (f<=BandF[i+1]))
            PMax[j,i] = np.max(Psd[fr])
    
    return np.sum(PMax[:,0])/tr,np.sum(PMax[:,1])/tr,np.sum(PMax[:,2])/tr,np.sum(PMax[:,3])/tr