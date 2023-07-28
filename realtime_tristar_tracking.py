# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:57:34 2023

@author: theja
"""
import matplotlib.pyplot as plt
import sounddevice as sd
import queue


input_audio_queue = queue.Queue()

def get_RME_USB(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'ASIO' in dev_name
        usb_in_name = 'USB' in dev_name
        if asio_in_name and usb_in_name:
            return i

usb_fireface_index = get_RME_USB(sd.query_devices())

block_size = 9182*3
fs = 44100
S = sd.InputStream(samplerate=44100, blocksize=4096, device=usb_fireface_index, channels=12)
S.start()

#%%
rawaudio, status = S.read(block_size)
audio = rawaudio[:,8:]


plt.figure()
plt.specgram(audio[:,0], Fs=fs)

