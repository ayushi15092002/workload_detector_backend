import mne
import os
from glob import glob
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

fname= "/content/drive/My Drive/mental workload/channel_loc.csv"
subjectPath = "/content/drive/My Drive/mental workload/subjects"

# read_dataset(subjectPath)

def read_dataset(subjectPath):
    
    try:
        root, dirs, files = next(os.walk(subjectPath))
        
        for folder_name in dirs:
          path_file_vhdr1 = subjectPath + '/' + folder_name + '/nback1.vhdr'
          path_file_vmrk1 = subjectPath + '/' + folder_name + '/nback1.vmrk'
          preprocess_data(path_file_vhdr1, path_file_vmrk1,fname, folder_name, "nback1")

          path_file_vhdr2 = subjectPath + '/' + folder_name + '/nback2.vhdr'
          path_file_vmrk2 = subjectPath + '/' + folder_name + '/nback2.vmrk'
          preprocess_data(path_file_vhdr2, path_file_vmrk2,fname, folder_name, "nback2")

          path_file_vhdr3 = subjectPath + '/' + folder_name + '/nback3.vhdr'
          path_file_vmrk3 = subjectPath + '/' + folder_name + '/nback3.vmrk'
          preprocess_data(path_file_vhdr3, path_file_vmrk3,fname,folder_name,"nback3")
            
    except StopIteration:
        pass
        print("Error ocurred:")
        print("Directory with dataset does not found!")
        print("Program will be terminated")
        exit(1)

def preprocess_data(path, mark, fname):

  TUB_montage = mne.channels.read_custom_montage(fname)
  raw = mne.io.read_raw_brainvision(path, eog=('HEOG', 'VEOG'), preload=True)
  raw.set_montage(TUB_montage)
  # raw.info
  # raw.plot(start=0, duration=6)
  mrk = mne.read_annotations(mark, sfreq='auto', uint16_codec=None)
  raw_filtered = raw.copy().filter(l_freq=0.1 , h_freq= 45)
  # raw_filtered.plot()
  raw_notch_filtered = raw_filtered.notch_filter(50, filter_length='auto', phase='zero')
  # raw_notch_filtered.plot()
  raw_re_referenced = mne.set_eeg_reference(raw_notch_filtered,ref_channels='average',copy=True, projection=False)
  finData, times = raw_re_referenced[:]
  # finData.plot()

  # fig, ax = plt.subplots(2)
  # raw.plot_psd(ax=ax[0], show = False, fmax = 60)
  # finData.plot_psd(ax=ax[1], show = False, fmax=60)
  # ax[0].set_title("PSD before filtering")
  # ax[1].set_title("PSD after filtering")
  # ax[1].set_xlabel('Frequency(Hz)')
  # fig.set_tight_layout(True)
  # plt.show()

  n_components = 10 #number of components you want to fit # can be either integer which typically implies number of channels - 1 (if applied average reference)
                    #if floating point number (0-1) fraction of total explained variance
  method = 'fastica'
  max_iter = 100
  fit_params = dict(fastica_it = 5)
  random_state = 42
  ica = mne.preprocessing.ICA(n_components = n_components,
                          method = method,
                          max_iter=max_iter,
                          random_state= random_state
                          )
  ica.fit(finData)

  finData.load_data()
  # ica.plot_sources(finData, show_scrollbars=False)
  # ica.plot_components(sphere=1)

  #Manual Eye Artifact Removal
  ica.exclude = [0, 3]  
  reconst_raw = finData.copy()
  ica.apply(reconst_raw)

  # finData.plot(title = "finData")
  # reconst_raw.plot(title = "Manual")

  #Automatic (Threshold Based) Eye Artifact Removal
  ica.exclude = []
  reconst_raw = finData.copy()
  # find which ICs match the EOG pattern
  eog_indices, eog_scores = ica.find_bads_eog(finData, threshold = 2.5)
  ica.exclude = eog_indices
  ica.apply(reconst_raw)

  #print(eog_indices)

  # reconst_raw.plot(title="Automatic")


  # # barplot of ICA component "EOG match" scores
  # ica.plot_scores(eog_scores)

  # # plot diagnostics
  # ica.plot_properties(finData, picks=eog_indices)

  # # plot ICs applied to raw data, with EOG matches highlighted
  # ica.plot_sources(finData, show_scrollbars=False)

  # # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
  # ica.plot_sources(eog_evoked)

  # reconst_raw.plot()

  events_ids = {
  'Stimulus/S 16': 0,
  'Stimulus/S 48': 2,
  'Stimulus/S 64': 2,
  'Stimulus/S 80': 3,
  'Stimulus/S 96': 3}
  events_ids

  event,event_ids = mne.events_from_annotations(reconst_raw, events_ids)

  # mne.viz.plot_events(event,event_id = events_ids, sfreq=raw.info['sfreq'])

  tmin=-0.3 # when does the epoch start relative to the event onset # 300ms before the start of the event
  tmax=1.7  # when does the event end after the even onset # 500 ms from the start of the event

  #Is a tuple containing the start of the baseline and end of the baseline
  baseline= (None, 0) #None mean begnining of the event and 0 is the start of the event 
  epochs = mne.Epochs(reconst_raw, 
                      events=event,
                      event_id=event_ids,
                      tmin=tmin,tmax=tmax, 
                      baseline=baseline,
                      preload=True,event_repeated = 'drop')

  # epochs.plot(events=event, event_id = event_ids)
  return epochs 
  # if session_name == "nback1":
  #   session_id = "1"
  # elif session_name == "nback2":
  #   session_id = "2"
  # else:
  #   session_id = "3"

  # try:
  #   # save the segmented epochs in the clean data directory 
  #   filepath = "/content/drive/My Drive/mental workload/epochs_data" + "/" +folder_name + "_"+session_id +".fif"
  #   mne.Epochs.save(epochs,fname=filepath,overwrite=True)
  # except FileExistsError:
  #   pass