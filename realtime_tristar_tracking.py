# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:57:34 2023

@author: theja
"""

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
#import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal 
from scipy.spatial.transform import Rotation

import queue
from common_functions import calc_rms,calc_multich_delays
from localisation_mpr2003 import tristar_mellen_pachter
np.random.seed(78464)

input_audio_queue = queue.Queue()

def get_RME_USB(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'ASIO' in dev_name
        usb_in_name = 'USB' in dev_name
        if asio_in_name and usb_in_name:
            return i
def get_60cm_2D_tristar_xyz():
    R = 0.6
    micxyz = np.zeros((4,3))
    micxyz[1,:] = [R*np.cos(60), 0, R*np.sin(30)]
    micxyz[2,:] = [-R*np.cos(60), 0, R*np.sin(30)]
    micxyz[3,:] = [0,0,R]
    micxyz += np.random.choice(np.linspace(1e-4,1e-3,100),micxyz.size).reshape(micxyz.shape)
    return micxyz

usb_fireface_index = get_RME_USB(sd.query_devices())
micxyz = get_60cm_2D_tristar_xyz()


#%%
tilt_deg = -21.0
# rotate the array on the x-axis a bit to the 'back' to mimic a realistic 
# positioning in the field. 
tilt_back = Rotation.from_euler('x',tilt_deg,degrees=True)
micxyz = micxyz.dot(tilt_back.as_matrix())

block_size = 4096
fs = 44100
S = sd.InputStream(samplerate=44100, blocksize=4096, device=usb_fireface_index, channels=12)
S.start()

#%%


audio_queue = queue.Queue()
source_solutions = queue.Queue()
threshold = 1e-2

vsound = 340.0 # m/s
highpass_coeffs = signal.butter(1, 15000/(fs*0.5), 'low')

def get_xyz_tristar():
    global audio_queue, source_solutions, chunknum, block_size
    
    buffer, status = S.read(block_size)
    audiochunk = buffer[:,8:12]
    audio_queue.put(buffer[:,8:12])
    if not audio_queue.empty():
        channel_rms  = np.array([calc_rms(audiochunk[:,each]) for each in range(audiochunk.shape[1])])
        if np.all(channel_rms>threshold):
            delays = calc_multich_delays(audiochunk, ba_filt=highpass_coeffs,fs=fs)
            di = delays*vsound
            solns = tristar_mellen_pachter(micxyz,di)
            #print('Miaow', solns)
            if len(solns)>0:
                source_solutions.put((solns[0]))
                return solns
    else:
        return np.array([])

#%%

app = pg.mkQApp("")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('Realtime angle-of-arrival plot')
w.setCameraPosition(distance=25)

g = gl.GLGridItem()
#g.setDepthValue(micxyz[2,2])
w.addItem(g)

# Add the microphone array here

mic_plot = gl.GLScatterPlotItem(pos=micxyz, color=(1,0,0,1), size=10)
w.addItem(mic_plot)

# Now add the source data as we get the audio
import collections 

all_sources = collections.deque(maxlen=5)
all_sources.append(np.array([100,100,100]))
source_plot = gl.GLScatterPlotItem(pos=all_sources, color=(1,1,0,1), size=10)
w.addItem(source_plot)
camdistance = 10
w.setCameraParams(distance=camdistance, azimuth=60, elevation=16)
# Also add a buffer number text label
# w.grabFramebuffer().save(f'only_array.png')

updatenum = 1

#%%

def update():
    global source_plot, all_sources, updatenum, w, chunknum, label, camdistance
    out = get_xyz_tristar()
    
    if out is None:
        pass
    elif len(out)==0:
        pass
    elif len(out)>0:
        updatenum += 1
        all_sources.append(out[0])
        source_plot.setData(pos=all_sources)

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(1)

if __name__ == '__main__':
    print('Remember to switch off any sound enhancement options in your OS!!\n \n')
    pg.exec()
