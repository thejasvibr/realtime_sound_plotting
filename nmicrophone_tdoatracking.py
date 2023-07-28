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
import glob
import natsort
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
from scipy.spatial.transform import Rotation

#%%
threed_tristar = True
micxyz = np.loadtxt('micxyz.csv', delimiter=',')

tilt_deg = -21.0
# rotate the array on the x-axis a bit to the 'back' to mimic a realistic 
# positioning in the field. 
tilt_back = Rotation.from_euler('x',tilt_deg,degrees=True)
micxyz = micxyz.dot(tilt_back.as_matrix())
# also add a height factor to include the tripod height. 
micxyz[:,2] += 1.57 # specific to 2019-05-07

orig_sources = np.loadtxt('sources.csv', delimiter=',')

audiotype = 'audiofile'
audiofile_path = './freefile_trista60cm.wav'
#audiofile_path = 'ACTxx_2019-05-07_21-30-57_0000100.WAV'
audiofile_path = 'ACTxx_2019-05-07_21-22-05_0000089.WAV'

fs = sf.info(audiofile_path).samplerate
loaded_durn = 9 # seconds
audio, fs = sf.read(audiofile_path, stop=int(fs*loaded_durn))

fs = sf.info(audiofile_path).samplerate
block_size = 2048
starts = range(0,audio.shape[0],int(block_size))

threshold = 1e-2
vsound = 340.0 # m/s
highpass_coeffs = signal.butter(1, 10000/(fs*0.5), 'high')
#%%
audio_queue = queue.Queue()
chunknum = 0
for i, startpoint in enumerate(starts):
    chunk = audio[startpoint:startpoint+block_size,:]
    audio_queue.put((chunk,i))
    
source_solutions = queue.Queue()

def get_xyz_tristar():
    global audio_queue, source_solutions, chunknum
    
    if not audio_queue.empty():
        audiochunk, chunknum = audio_queue.get()
        print(chunknum, w.cameraParams()['azimuth'])
        channel_rms  = np.array([calc_rms(audiochunk[:,each]) for each in range(audio.shape[1])])
        if np.all(channel_rms>threshold):
            delays = calc_multich_delays(audiochunk, ba_filt=highpass_coeffs,fs=fs)
            di = delays*vsound
            solns = tristar_mellen_pachter(micxyz,di)
            if len(solns)>0:
                source_solutions.put((solns,chunknum))
                return solns
    else:
        return np.array([])

#%%


app = pg.mkQApp("Realtime angle-of-arrival plot")
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

all_sources = []
all_sources.append(np.array([100,100,100]))
source_plot = gl.GLScatterPlotItem(pos=all_sources, color=(1,1,0,1), size=10)
w.addItem(source_plot)
camdistance = 20
w.setCameraParams(distance=camdistance, azimuth=0)
# Also add a buffer number text label
w.grabFramebuffer().save(f'only_array.png')

updatenum = 0

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
        for each in out:
            all_sources.append(each)
        source_plot.setData(pos=all_sources)
        w.grabFramebuffer().save(f'fileName_{chunknum}.png')
        if w.cameraParams()['azimuth']>=-90:
            w.orbit(-2,-0.25)
        else:
            camdistance += 0.25
            w.setCameraParams(distance=camdistance)
            w.orbit(0,2)

    #time.sleep(0.02)

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(1)

if __name__ == '__main__':
    print('Remember to switch off any sound enhancement options in your OS!!\n \n')
    pg.exec()
    # And then collect all the .png files
    image_files = natsort.natsorted(glob.glob('fileName*.png'))
    from PIL import Image, ImageDraw, ImageFont
    
    # Create the frames
    frames = []
    all_solns = []
    for img in image_files:
        new_frame = Image.open(img)       
        draw = ImageDraw.Draw(new_frame)
        solns, chunknum = source_solutions.get()
        draw.text((0, 0), f"Buffer number {chunknum}",(255,0,255))
        frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    frames[0].save('png_to_gif.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=80, loop=0)    
    import os 
    [os.remove(each) for each in image_files]
        