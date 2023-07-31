# -*- coding: utf-8 -*-
"""
Live stream angle of arrival with 2 laptop microphones
======================================================
The script calculates the inter-mic delay and gives an idea of the
angle of arrival of sound.

How to run this script
----------------------
Start your conda/venv and in the command line type
>>> python realtime_timedelay_vis.py

A black PyQtGraph window should pop-up and begin displaying the real-time
estimated time-delay from the laptop's stereo audio.

Which kinds of sounds to try out
--------------------------------
> Claps
> Whistles
> White noise with your mouth (shhhhhh)
> Download a function generator app on your phone and try out which of these signal
types is reliably tracked. 
    > White noise
    > Tonal sounds from low -> high frequency - what happens? 
> Try talking with
    


What parameters to change specific to your laptop
-------------------------------------------------
* Alter the ```bp_freq``` variable to allow difference min-max frequencies
* Alter the ```threshold``` to a lower values if the time-delay vis doesn't move at all, 
and to a higher value if it;s moving all the time. 

Author:Thejasvi Beleyur
License: MIT License
grids and plot layout based off the example scripts in the pyqtgraph library
"""

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from scipy import signal 
import sounddevice as sd

def calc_rms(in_sig):
    '''
    
    '''
    rms_sig = np.sqrt(np.mean(in_sig**2))
    return(rms_sig)

def calc_delay(two_ch,ba_filt,fs=44100):
    '''
    Parameters
    ----------
    two_ch : (Nsamples, 2) np.array
        Input audio buffer
    ba_filt : (2,) tuple
        The coefficients of the low/high/band-pass filter
    fs : int, optional
        Frequency of sampling in Hz. Defaults to 44.1 kHz
    
    Returns
    -------
    delay : float
        The time-delay in seconds between the arriving audio across the 
        channels. 
    '''
    for each_column in range(2):
        two_ch[:,each_column] = signal.lfilter(ba_filt[0],ba_filt[1],two_ch[:,each_column])

    cc = np.correlate(two_ch[:,1],two_ch[:,0],'same')
    midpoint = cc.size/2.0
    delay = np.argmax(cc) - midpoint
    delay *= 1/float(fs)

    return delay


app = pg.mkQApp("Realtime angle-of-arrival plot")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('Realtime angle-of-arrival plot')
w.setCameraPosition(distance=25)

g = gl.GLGridItem()
w.addItem(g)




mypos = np.random.normal(0,1,size=(20,3))*5
mypos[:,2] = np.abs(mypos[:,2])
sp_my = gl.GLScatterPlotItem(pos=mypos, color=(1,1,1,1), size=10)
w.addItem(sp_my)

#%% Set up the audio-stream of the laptop, along with how the 
# incoming audio buffers will be processed and thresholded.
fs = 48000
block_size = 4096

bp_freq = np.array([10,10000.0]) # the min and max frequencies
# to be 'allowed' in Hz.

ba_filt = signal.butter(2, bp_freq/float(fs*0.5),'bandpass')

S = sd.InputStream(samplerate=fs,blocksize=block_size,channels=2, latency='low')
S.start()


all_xs = np.linspace(-10,10,S.blocksize)
threshold = 1e-2


guidepos = np.column_stack((all_xs, np.zeros(S.blocksize), np.zeros(S.blocksize)))
guideline = gl.GLScatterPlotItem(pos=guidepos, color=(1,0,1,1), size=10)
w.addItem(guideline)


def update():
    global sp_my, all_xs, threshold, S, ba_filt
    
    try:
        in_sig,status = S.read(S.blocksize)
        delay_crossch = calc_delay(in_sig,ba_filt,fs)
        rms_sig = calc_rms(in_sig[:,0])
        if rms_sig > threshold:
            movement_amp_factor = 3e4
            all_zs = np.tile(rms_sig*movement_amp_factor*1e-3, S.blocksize)
            all_delay = np.tile(-delay_crossch*movement_amp_factor,
                                 S.blocksize)
            all_ys = in_sig[:,0]+all_delay
            xyz = np.column_stack((all_xs,all_ys,all_zs))
        else:
            # when there's no/low signal at the mics
            y = np.zeros(S.blocksize)
            z= y.copy()
            xyz = np.column_stack((all_xs,y,z))
      
        sp_my.setData(pos=xyz)
    except KeyboardInterrupt:
        S.stop()

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(5)

if __name__ == '__main__':
    print('Remember to switch off any sound enhancement options in your OS!!')
    pg.exec()