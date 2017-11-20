import tensorflow as tf
import numpy as np

import time
import sys

import threading
import datetime
#import pyaudio
#from scikits.talkbox.features import mfcc
#from matplotlib import pyplot as plt
#from matplotlib import cm
#from PIL import Image
#import wave



### Parameter definition
# Data files
DATADIR = 'data/'
DATAFILES = ['dryer-hot1.txt','dryer-low1.txt','fan.txt','heater300.txt','microwave1.txt','none.txt','led.txt','dryer-cool1.txt']
LABELS = ['dryer-high','dryer-low','fan','heater','microwave', 'OFF', 'LED','dryer-cool']

# Neural network
SAMPLES=200

TRAIN_NUM=200
TEST_NUM=50
REC_NUM=1

INPUT_NUM=50
INPUT1_NUM=25
HIDDEN_NUM=100
OUTPUT_NUM=len(DATAFILES)


# Data aquisition
port = 26
interval = 0.0002
samples = 50000

# SPI
def readAdc():
    adc = spi.xfer2([0x06,0,0])
    data1 = ((adc[1]&0x0f) << 8) + adc[2]
    adc = spi.xfer2([0x06,0x40,0])
    data2 = ((adc[1]&0x0f) << 8) + adc[2]
    return data1, data2

def convertVoltage(data):
    volts = (data * 5.0) / float(1023)
    volts = round(volts,4)
    return volts

def convertCurrent(data):
    curr = (data - 512)  / 512.0 * 2.5 / 0.10416
    curr = round(curr,4)
    return curr


dataset = np.empty((0,INPUT_NUM), np.float32)
testdataset = np.empty((0,INPUT_NUM), np.float32)
ydata = np.empty((0,OUTPUT_NUM), np.float32)
testydata = np.empty((0,OUTPUT_NUM), np.float32)
dataset_1 = np.empty((0,INPUT1_NUM), np.float32)
testdataset_1 = np.empty((0,INPUT1_NUM), np.float32)
dataset_2 = np.empty((0,INPUT1_NUM), np.float32)
testdataset_2 = np.empty((0,INPUT1_NUM), np.float32)
testdataset_3 = np.empty((0,INPUT_NUM), np.float32)
testdataset_4 = np.empty((0,INPUT1_NUM), np.float32)
testdataset_5 = np.empty((0,INPUT_NUM), np.float32)




data = np.genfromtxt('C:/Users/souta uehara/Documents/powerrecog/data/LED+fan.txt', delimiter="\t", dtype= int)[:,2] - (2**11)

#real part
r = np.real(np.fft.fft(
  data[0:SAMPLES*TRAIN_NUM].reshape([TRAIN_NUM,SAMPLES])
  )[:,0:INPUT1_NUM]).astype(np.float32)

dataset_1 = np.append(dataset_1, r, axis = 0)



I = np.imag(np.fft.fft(
  data[0:SAMPLES*TRAIN_NUM].reshape([TRAIN_NUM,SAMPLES])
  )[:,0:INPUT1_NUM]).astype(np.float32)

dataset_2 = np.append(dataset_2, I, axis = 0)



dataset = np.hstack((dataset_1 , dataset_2))
dataset_1 = dataset.reshape([10000])

np.savetxt('LED+fan_realimag_data.csv', dataset_1, delimiter=',')




#imaginaly
