# -*- coding: utf-8 -*-
"""
DOA of a bird calling - recorded with a 2 channel device. 
=========================================================

Getting similar data
~~~~~~~~~~~~~~~~~~~~
Go to Xeno-Canto and type any combination of these in the search box:
    >>> smp:">22050" q:A dvc:"pcm-m10" 
    >>> smp:">22050" q:A dvc:"pcm-m10" type:duet 
    
Click on the Cat.Nr. in the results list and check to see that the file is a 
WAV file. If so, then proceed to download the fiel and place it into the working
directory. Here in this example - we're working on the XC819675 Catalogue Number
recording (you can also just put this number into the search box.)

Author: Thejasvi Beleyur
License: MIT license.
"""

import numpy as np 
import pyroomacoustics as pra
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as signal 
import scipy.ndimage as ndi

#%% Define the distance between microphones
# https://www.sony.com.sg/electronics/support/digital-voice-recorders-pcm-series/pcm-m10/specifications
mic_separation = 62e-3 # metres

#%% Load the manually selected portions of the audio file and do some basic filtering.
tstarts = np.array([0.585, 7.063, 8.677, 23.778, 25.998])
tstops = tstarts+0.06
time_windows = np.column_stack((tstarts, tstops))


tstart, tstop = time_windows[2,:]# choose your time window of interest here. 
filename = 'XC819675 - Identity unknown.wav'
fs = sf.info(filename).samplerate
audio, fs = sf.read(filename)

b,a = signal.butter(2, np.array([1e3,7e3])/(fs*0.5), 'bandpass')
audio = np.apply_along_axis(lambda X: signal.filtfilt(b,a,X), 0, audio)
#%% Now much to be seen here. Let's try the Steered-Response-Power Phase alogirhtm -
# which 
start_t, stop_t = np.int32(fs*time_windows[0,:])

audio_segment = audio[start_t:stop_t,:]
micxyz = np.array([[0,0,0],
                  [-62e-3,0,0]]).T
nfft = 512
X = pra.transform.stft.analysis(
     audio_segment, nfft, nfft // 2, win=np.hanning(nfft)
 )
X = np.swapaxes(X, 2, 0)

srp_phat = pra.doa.SRP(micxyz, fs, nfft) # initialise an SRP-instance for your data
srp_phat.locate_sources(X)

print(f'Angle of arrival is: {srp_phat.azimuth_recon}')

#%% Now let's check to see if there's any deviation in the angle-of-arrival
# across the various calls detected. 
for timestamps in time_windows:
    start_t, stop_t = np.int32(fs*timestamps)

    cleaned_audio = audio[start_t:stop_t,:]
    micxyz = np.array([[0,0,0],
                      [-62e-3,0,0]]).T
    nfft = 512
    X = pra.transform.stft.analysis(
         cleaned_audio, nfft, nfft // 2, win=np.hanning(nfft)
     )
    X = np.swapaxes(X, 2, 0)


    srp_phat = pra.doa.SRP(micxyz, fs, nfft) # initialise an SRP-instance for your data
    srp_phat.locate_sources(X)

    print(f'Angle of arrival is: {srp_phat.azimuth_recon}')



#%%
# Scaling it up!!
# ~~~~~~~~~~~~~~~
# Defining time-windows manually is too cumbersome - let's just
# automate the thresholding to get *all* the segments where there's a call
# above a threshold by getting the Hilbert envelope of the first channel
hilbert_env = np.abs(signal.hilbert(audio[:,0]))
# smooth the envelope a bit
smoothing_size = 0.05
smoother = np.ones(int(fs*smoothing_size))/int(fs*smoothing_size)
hilber_env_smooth = signal.convolve(hilbert_env, smoother)


plt.figure()
plt.plot(hilber_env_smooth)


threshold = 10e-3
above_threshold = hilber_env_smooth>threshold
plt.hlines(threshold, 0, hilber_env_smooth.size,'r')

# detect all continuous patches above threshold
chunks_abovethresh, num_calls = ndi.label(above_threshold)
separate_calls = ndi.find_objects(chunks_abovethresh)

calls = []
min_samples = int(fs*0.05)
for each in separate_calls:
    detn_durn = each[0].stop-each[0].start
    if detn_durn>=min_samples:
        calls.append(audio[each[0],:])

#%%
# How stable is the angle of arrival across time. 

call_aoa = []
for call in calls:
    nfft = 256
    X = pra.transform.stft.analysis(
         call, nfft, nfft // 2, win=np.hanning(nfft)
     )
    X = np.swapaxes(X, 2, 0)


    srp_phat = pra.doa.SRP(micxyz, fs, nfft) # initialise an SRP-instance for your data
    srp_phat.locate_sources(X,freq_range=[2000,8000])
    call_aoa.append(srp_phat.azimuth_recon)
    #print(f'Angle of arrival is: {srp_phat.azimuth_recon}')

plt.figure()
plt.plot(call_aoa, '*-')
plt.ylim(-20,20)
plt.grid()
plt.ylabel('Angle of arrival (azimuth, degrees)', fontsize=12)
plt.xlabel('Detected call #', fontsize=12)


#%% Let's try to pick up the time-delay of arrival between the two microphones. 
# This doesn't work so well as the mics are so close together, w.r.t the 
# sound's wavelength.

crosscorr = signal.correlate(audio[:,0], audio[:,1])
plt.figure()
plt.plot(crosscorr)
plt.vlines(crosscorr.size*0.5, 0,crosscorr.max(),'r')



