# -*- coding: utf-8 -*-
"""begins an audio stream and plots some feature of the audio stream
as a 3d plot in real time ..or so..
Created on Sat Dec 02 20:36:53 2017

@author: tbeleyur
with help setting up the plot from the Pyqtgraph example scripts
"""
import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sounddevice as sd




def calc_rms(in_sig):
    rms_sig = np.sqrt(np.mean(in_sig**2))
    return(rms_sig)

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
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

p = np.array([0,0,0]).reshape((1,3))
all_colours = np.ones((1,4))
all_sizes = np.ones(p.shape[0])
pl = gl.GLScatterPlotItem(pos=p,color=all_colours)
w.addItem(pl)

S = sd.Stream(192000,blocksize=2048,channels=1)

all_xs = np.linspace(-1,1,S.blocksize)
#all_ys = all_xs
all_colors = np.ones(S.blocksize*4).reshape((-1,4))
S.start()

def update():
    global S,calc_rms,all_xs, all_ys, all_colors

    try:
        in_sig,status = S.read(S.blocksize)
        rms_sig = calc_rms(in_sig)
        all_z = np.tile(rms_sig,S.blocksize)
        xyz = np.column_stack((all_xs,in_sig,all_z))

        pl.setData(pos=xyz,color=all_colors)
    except KeyboardInterrupt:
        S.stop()
        sys.exit()




t = QtCore.QTimer()
t.timeout.connect(update)
t.start(0)


