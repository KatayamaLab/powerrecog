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
DATADIR = 'data/'


data_LED = np.genfromtxt('/Users/souta uehara/Documents/powerrecog/data/led.txt', delimiter="\t", dtype= int)[:,2] - (2048)
data_fan = np.genfromtxt('/Users/souta uehara/Documents/powerrecog/data/fan.txt', delimiter="\t", dtype= int)[:,2] - (2048)
data_heater = np.genfromtxt('/Users/souta uehara/Documents/powerrecog/data/heater300.txt', delimiter="\t", dtype= int)[:,2] - (2048)
data_microwave = np.genfromtxt('/Users/souta uehara/Documents/powerrecog/data/microwave1.txt', delimiter="\t", dtype= int)[:,2] - (2048)
data_dryer = np.genfromtxt('/Users/souta uehara/Documents/powerrecog/data/dryer-low1.txt', delimiter="\t", dtype= int)[:,2] - (2048)
data_none = np.genfromtxt('/Users/souta uehara/Documents/powerrecog/data/none.txt', delimiter="\t", dtype= int)[:,2] - (2048)
data_heaterdryer = np.genfromtxt('/Users/souta uehara/Documents/powerrecog/data/heater+dryer-low.txt', delimiter="\t", dtype= int)[:,2] - (2048)
data_microwaveheater = np.genfromtxt('/Users/souta uehara/Documents/powerrecog/data/heater+microwave.txt', delimiter="\t", dtype= int)[:,2] - (2048)
data_microwavedryer = np.genfromtxt('/Users/souta uehara/Documents/powerrecog/data/microwave+dryer-low.txt', delimiter="\t", dtype= int)[:,2] - (2048)

data_microwave_LED = data_microwave + data_LED
data_microwave_fan = data_microwave + data_fan
data_microwave_heater = data_microwave + data_heater
data_microwave_dryer = data_microwave + data_dryer
data_heater_LED = data_heater + data_LED
data_heater_fan = data_heater + data_fan
data_heater_dryer = data_heater + data_dryer
data_dryer_LED = data_dryer + data_LED
data_dryer_fan = data_dryer + data_fan
data_LED_fan = data_LED + data_fan

DATAFILES =  [data_dryer, data_LED, data_fan, data_heater, data_microwave, data_none, data_microwave_LED, data_microwave_fan, data_microwave_heater, data_microwave_dryer, data_heater_LED, data_heater_fan, data_heater_dryer, data_dryer_LED, data_dryer_fan, data_LED_fan]
LABELS = ['dryer','LED','fan','heater','microwave','OFF', 'microwave+LED', 'microwave+fan', 'microwave+heater', 'microwave+dryer', 'heater+LED', 'heater+fan', 'heater+dryer', 'dryer+LED', 'dryer+fan', 'LED+fan']

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


dataset = np.empty((0,INPUT_NUM), np.float32)
testdataset = np.empty((0,INPUT1_NUM), np.float32)
ydata = np.empty((0,OUTPUT_NUM), np.float32)
testydata = np.empty((0,OUTPUT_NUM), np.float32)
dataset_1 = np.empty((0,INPUT1_NUM), np.float32)
testdataset_1 = np.empty((0,INPUT1_NUM), np.float32)
dataset_2 = np.empty((0,INPUT1_NUM), np.float32)
testdataset_2 = np.empty((0,INPUT1_NUM), np.float32)

data = data_microwavedryer

t = np.abs(np.fft.fft(
    data[0:SAMPLES*TRAIN_NUM].reshape([TRAIN_NUM,SAMPLES])
    )[:,0:INPUT_NUM]).astype(np.float32)

dataset = np.append(dataset, t, axis = 0)





np.savetxt('microwave+dryer_act.txt', dataset, delimiter="\t", newline="\t")
