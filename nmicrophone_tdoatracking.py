# -*- coding: utf-8 -*-
"""Live stream acoustic tracking with N microphones
The script calculates the inter-mic delay and gives an idea of the
angle of arrival of sound.


Currently tested to work with the following Python and package versions:
    * Python 3.10.0 (conda-forge)
    * NumPy 1.25.1
    * PyQtGraph 0.13.3
    * Scipy 1.11.1
    * sounddevice 0.4.6

Created on Wed July 26 2023

@author:tbeleyur
grids and plot layout based off the example scripts in the pyqtgraph library
"""

import numpy as np
import time
import queue
from queue import Empty
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from scipy import signal 
import sounddevice as sd
import soundfile as sf
from common_functions import calc_rms, calc_multich_delays
from localisation_mpr2003 import tristar_mellen_pachter
#%%
micxyz = np.loadtxt('micxyz.csv', delimiter=',')
orig_sources = np.loadtxt('sources.csv', delimiter=',')

audiotype = 'audiofile'
audiofile_path = './freefile_trista60cm.wav'

audio, fs = sf.read(audiofile_path)

fs = sf.info(audiofile_path).samplerate
block_size = 4096
starts = range(0,audio.shape[0],int(block_size))

threshold = 1e-2
vsound = 340.0 # m/s
highpass_coeffs = signal.butter(1, 100/(fs*0.5), 'high')
#%%
audio_queue = queue.Queue()

for i, startpoint in enumerate(starts):
    chunk = audio[startpoint:startpoint+block_size,:]
    audio_queue.put(chunk)
    


def get_xyz_tristar():
    global audio_queue
    
    if not audio_queue.empty():
        audiochunk = audio_queue.get()
        channel_rms  = np.array([calc_rms(audiochunk[:,each]) for each in range(audio.shape[1])])
        if np.all(channel_rms>threshold):
            delays = calc_multich_delays(audiochunk, ba_filt=highpass_coeffs)
            di = delays*vsound
            solns = tristar_mellen_pachter(micxyz,di)
            if solns[0].size>0:
                return solns[0]
    else:
        return np.array([])
# import time
# start = time.time()
# for i in range(10):
#     get_xyz_tristar()
# stop = time.time()
#print(f'Time elapsed for 10 runs: {stop-start}')
#%%


app = pg.mkQApp("Realtime angle-of-arrival plot")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('Realtime angle-of-arrival plot')
w.setCameraPosition(distance=25)

g = gl.GLGridItem()
g.setDepthValue(micxyz[2,2])
w.addItem(g)

# Add the microphone array here

mic_plot = gl.GLScatterPlotItem(pos=micxyz, color=(1,0,0,1), size=10)
w.addItem(mic_plot)

# Now add the source data as we get the audio

all_sources = []
all_sources.append(micxyz[0,:])
source_plot = gl.GLScatterPlotItem(pos=all_sources, color=(1,1,0,1), size=10)
w.addItem(source_plot)

def update():
    global source_plot, all_sources
    out = get_xyz_tristar()
    if out is None:
        pass
    elif out.size>0:
        all_sources.append(out)
        source_plot.setData(pos=all_sources)
    time.sleep(0.2)
    
    

    
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(5)

if __name__ == '__main__':
    print('Remember to switch off any sound enhancement options in your OS!!')
    pg.exec()