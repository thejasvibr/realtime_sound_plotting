# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:00:50 2023

@author: theja
"""
import matplotlib.pyplot as plt
import numpy as np 
import pyroomacoustics as pra
import scipy.signal as signal 
from scipy.spatial.transform import Rotation
import soundfile as sf
np.random.seed(78464)
# Make a small room simulation with the tristar array (60 cm)
R = 0.6
micxyz = np.zeros((4,3))
micxyz[0,1] += 0.3
micxyz[1,:] = [R*np.cos(60), 0, R*np.sin(30)]
micxyz[2,:] = [-R*np.cos(60), 0, R*np.sin(30)]
micxyz[3,:] = [0,0,R]
micxyz += np.random.choice(np.linspace(1e-4,1e-3,100),micxyz.size).reshape(micxyz.shape)





# Create a free-field simulation
fs = 192000
freefield = pra.AnechoicRoom(  
             fs=fs)
freefield.add_microphone_array(micxyz.T)

# generate random points at various distances 

t = np.linspace(0,2048/fs,2048)

source_sound = signal.chirp(t,15000,t[-1],5000,'log')
source_sound *= signal.windows.tukey(source_sound.size, alpha=0.95)

n_sources = 20
t = np.linspace(0,2,n_sources)
x,y,z = 3*np.cos(2*np.pi*0.2*t), 2*np.sin(2*np.pi*0.2*t), np.tile(0.3,t.size)
source_points = np.column_stack((x,y,z))
source_points[:,1] = np.abs(source_points[:,1])

# source_points = np.random.random(n_sources*3).reshape(-1,3)*10
# source_points[:,1] = np.abs(source_points[:,1])
for i,each in enumerate(source_points):
    freefield.add_source(each, signal=source_sound, delay=i*0.1)

# run  simulation 
freefield.simulate()
#%%
plt.figure()
plt.specgram(freefield.mic_array.signals[1,:],Fs=fs)

sf.write('freefile_trista60cm.wav', freefield.mic_array.signals.T, samplerate=fs)

np.savetxt('micxyz.csv', micxyz,delimiter=',')
np.savetxt('sources.csv', source_points,delimiter=',')

