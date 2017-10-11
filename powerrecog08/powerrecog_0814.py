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
DATAFILES = ['dryer-hot1.txt','dryer-low1.txt','fan.txt','heater300.txt','microwave1.txt','none.txt','led.txt','dryer-cool1.txt','microwave+heater.txt','microwave+LED.txt','microwave+fun.txt','microwave+dryer-high.txt','microwave+dryer-low.txt','heater+LED.txt','heater+microwave.txt','heater+fun.txt','heater+dryer-high.txt','heater+dryer-low.txt']
LABELS = ['dryer-high','dryer-low','fan','heater','microwave', 'OFF', 'LED','dryer-cool','microwave+heater','microwave+LED','microwave+fun','microwave+dryer-high','microwave+dryer-low','heater+LED','heater+microwave','heater+fun','heater+dryer-high','heater+dryer-low']

# Neural network
SAMPLES=200

TRAIN_NUM=150
TEST_NUM=100
REC_NUM=1

INPUT_NUM=200
INPUT1_NUM=100
HIDDEN_NUM=100
OUTPUT_NUM=len(DATAFILES)


# Data aquisition
port = 26
interval = 0.0002
samples = 50000


dataset = np.empty((0,INPUT1_NUM), np.float32)
testdataset = np.empty((0,INPUT1_NUM), np.float32)
ydata = np.empty((0,OUTPUT_NUM), np.float32)
testydata = np.empty((0,OUTPUT_NUM), np.float32)
dataset_1 = np.empty((0,INPUT1_NUM), np.float32)
testdataset_1 = np.empty((0,INPUT1_NUM), np.float32)
dataset_2 = np.empty((0,INPUT1_NUM), np.float32)
testdataset_2 = np.empty((0,INPUT1_NUM), np.float32)




for i, filename in enumerate(DATAFILES):
  data = np.genfromtxt(DATADIR + filename, delimiter="\t", dtype= int)[:,2] - (2**11)

#real part

  r = np.real(np.fft.fft(
    data[0:SAMPLES*TRAIN_NUM].reshape([TRAIN_NUM,SAMPLES])
    )[:,0:INPUT1_NUM]).astype(np.float32)

  dataset_1 = np.append(dataset_1, r, axis = 0)

  r = np.real(np.fft.fft(
    data[SAMPLES*TRAIN_NUM:SAMPLES*(TRAIN_NUM+TEST_NUM)].reshape([TEST_NUM,SAMPLES])
    )[:,0:INPUT1_NUM]).astype(np.float32)

  testdataset_1 = np.append(testdataset_1, r, axis = 0)

  r = np.zeros([TRAIN_NUM, OUTPUT_NUM])
  r[:,i] = 1
  ydata = np.append(ydata, r, axis = 0)

  r = np.zeros([TEST_NUM, OUTPUT_NUM])
  r[:,i] = 1
  testydata = np.append(testydata, r, axis = 0)

#imaginaly


  I = np.imag(np.fft.fft(
    data[0:SAMPLES*TRAIN_NUM].reshape([TRAIN_NUM,SAMPLES])
    )[:,0:INPUT1_NUM]).astype(np.float32)

  dataset_2 = np.append(dataset_2, I, axis = 0)

  I = np.imag(np.fft.fft(
    data[SAMPLES*TRAIN_NUM:SAMPLES*(TRAIN_NUM+TEST_NUM)].reshape([TEST_NUM,SAMPLES])
    )[:,0:INPUT1_NUM]).astype(np.float32)

  testdataset_2 = np.append(testdataset_2, I, axis = 0)



dataset = np.hstack((dataset_1 , dataset_2))
testdataset = np.hstack((testdataset_1 , testdataset_2))

print(dataset_1)
print(dataset_2)
