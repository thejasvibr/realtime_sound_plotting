# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:53:33 2023

@author: theja
"""
import numpy as np 
import scipy.signal as signal 


def calc_rms(in_sig):
    rms_sig = np.sqrt(np.mean(in_sig**2))
    return(rms_sig)

def calc_delay(two_ch,ba_filt,fs=44100):

    for each_column in range(2):
        two_ch[:,each_column] = signal.lfilter(ba_filt[0],ba_filt[1],two_ch[:,each_column])

    cc = np.correlate(two_ch[:,1],two_ch[:,0],'same')
    midpoint = cc.size/2.0
    delay = np.argmax(cc) - midpoint
    delay *= 1/float(fs)

    return delay

def calc_multich_delays(multich_audio, **kwargs):
    '''
    Calculates peak delay based with reference ot 
    channel 1. 
    '''
    nchannels = multich_audio.shape[1]
    delay_set = []
    for each in range(1, nchannels):
        
        delay_set.append(calc_delay(multich_audio[:,[0,each]],**kwargs))
    return delay_set