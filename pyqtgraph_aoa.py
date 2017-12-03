# -*- coding: utf-8 -*-
"""Live stream angle of arrival with 2 laptop microphones
The script calculates the inter-mic delay and gives an idea of the
angle of arrival of sound.

Created on Wed Nov 29 10:51:24 2017

@author:tbeleyur
grids and plot layout based off the example scripts in the pyqtgraph library
"""
import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sounddevice as sd
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

    return(delay)


app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 1
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')


xgrid = gl.GLGridItem()
ygrid = gl.GLGridItem()
zgrid = gl.GLGridItem()
w.addItem(xgrid)
w.addItem(ygrid)
w.addItem(zgrid)
## rotate x and y grids to face the correct direction
xgrid.rotate(90, 0, 1, 0)
ygrid.rotate(90, 1, 0, 0)

## scale each grid differently
xgrid.scale(0.1,0.1,1)
ygrid.scale(0.1,0.1,1)
zgrid.scale(0.1,0.1,1)



fs = 96000
block_size = 4096
S = sd.Stream(samplerate=fs,blocksize=block_size,channels=2)
S.start()

hp_freq = 100.0
ba_filt = signal.butter(2,hp_freq/float(fs),'high')


p = np.zeros(block_size*3).reshape((-1,3))
all_colors = np.ones(block_size*4).reshape((-1,4))
pl = gl.GLScatterPlotItem(pos=p,color=all_colors)
w.addItem(pl)

all_xs = np.linspace(-1,1,S.blocksize)


def update():
    global S,calc_rms,all_xs, all_colors, fs

    try:
        in_sig,status = S.read(S.blocksize)
        delay_crossch = calc_delay(in_sig,ba_filt,fs)
        rms_sig = calc_rms(in_sig[:,0])
        all_zs = np.tile(rms_sig,S.blocksize)
        all_delay = np.tile(-delay_crossch*10**4,S.blocksize)
        all_ys = in_sig[:,0]+all_delay
        xyz = np.column_stack((all_xs,all_ys,all_zs))

        pl.setData(pos=xyz,color=all_colors)
    except KeyboardInterrupt:
        S.stop()
        sys.exit()



t = QtCore.QTimer()
t.timeout.connect(update)
t.start(0)
