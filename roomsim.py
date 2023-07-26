# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:00:50 2023

@author: theja
"""
import matplotlib.pyplot as plt
import numpy as np 
import pyroomacoustics as pra
import scipy.signal as signal 
np.random.seed(78464)
# Make a small room simulation with the tristar array (60 cm)
R = 0.6
micxyz = np.zeros((4,3))
micxyz[1,:] = [R*np.cos(60), 0, R*np.sin(30)]
micxyz[2,:] = [-R*np.cos(60), 0, R*np.sin(30)]
micxyz[3,:] = [0,0,R]




# Create a free-field simulation
fs = 44100
freefield = pra.AnechoicRoom(  
             fs=fs)
freefield.add_microphone_array(micxyz.T)

# generate random points at various distances 

t = np.linspace(0,2048/fs,2048)

source_sound = signal.chirp(t,15000,t[-1],5000,'log')
source_sound *= signal.windows.tukey(source_sound.size, alpha=0.95)

n_sources = 10
source_points = np.random.random(n_sources*3).reshape(-1,3)*10
for i,each in enumerate(source_points):
    freefield.add_source(each, signal=source_sound, delay=i*0.1)

# run  simulation 
freefield.simulate()
#%%
plt.figure()
plt.specgram(freefield.mic_array.signals[1,:],Fs=fs)



