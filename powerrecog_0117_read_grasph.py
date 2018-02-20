import tensorflow as tf
import numpy as np

import time
import sys
import spidev
import RPi.GPIO as GPIO
import threading
import datetime
import matplotlib.pyplot as plt
#import pyaudio
#from scikits.talkbox.features import mfcc
#from matplotlib import pyplot as plt
#from matplotlib import cm
#from PIL import Image
#import wave

### Parameter definition
# Data files
DATADIR = 'data/'

DATAFILES =  ['dryer-low1.txt','led.txt','fan.txt','heater300.txt']

LABELS = ['dryer','LED','fan','heater','microwave','OFF']

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




for i, filename in enumerate(DATAFILES):

  data = np.genfromtxt(DATADIR + filename, delimiter="\t", dtype= int)[:,2] - (2**11)



  t = np.abs(np.fft.fft(
    data[0:SAMPLES*TRAIN_NUM].reshape([TRAIN_NUM,SAMPLES])
    )[:,0:INPUT_NUM]).astype(np.float32)

  dataset = np.append(dataset, t, axis = 0)

  t = np.abs(np.fft.fft(
    data[SAMPLES*TRAIN_NUM:SAMPLES*(TRAIN_NUM+TEST_NUM)].reshape([TEST_NUM,SAMPLES])
    )[:,0:INPUT_NUM]).astype(np.float32)

  testdataset = np.append(testdataset, t, axis = 0)

  t = np.zeros([TRAIN_NUM, OUTPUT_NUM])
  t[:,i] = 1
  ydata = np.append(ydata, t, axis = 0)

  t = np.zeros([TEST_NUM, OUTPUT_NUM])
  t[:,i] = 1
  testydata = np.append(testydata, t, axis = 0)




#dataset = dataset/np.mean(dataset)
#testdataset = testdataset/np.mean(testdataset)

#print(dataset)


def weight_variable(shape):
    #"""適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
    #"""

    initial =  tf.truncated_normal(shape, stddev=0.14)
    return tf.Variable(initial)

def weight_variable1(shape):
    #"""適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
    #"""

    initial =  tf.truncated_normal(shape, stddev=0.16)
    return tf.Variable(initial)

def weight_variable2(shape):
    #"""適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
    #"""

    initial =  tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial)

def weight_variable3(shape):
    #"""適度にノイズを含んだ（対称性の除去と勾配ゼロ防止のため）重み行列作成関数
    #"""

    initial =  tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial)




def bias_variable(shape):
    #"""バイアス行列作成関数
    #"""
   initial = tf.truncated_normal(shape, stddev=0.14)
   return tf.Variable(initial)



keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, INPUT_NUM]) # None... any length
W1 = weight_variable([INPUT_NUM, HIDDEN_NUM]) # weight
b1 = bias_variable([HIDDEN_NUM]) # bias
a1 = tf.nn.relu(tf.matmul(x, W1) + b1) # output
W2 = weight_variable1([HIDDEN_NUM, 75]) # weight
b2 = bias_variable([75]) # bias
a2 = tf.nn.relu(tf.matmul(a1, W2) + b2) # output
W3 = weight_variable2([75, 50]) # weight
b3 = bias_variable([50]) # bias
a3 = tf.nn.relu(tf.matmul(a2, W3) + b3) # output
a3_drop = tf.nn.dropout(a3, keep_prob)
W4 = weight_variable3([50, 25]) # weight
b4 = bias_variable([25]) # bias
a4 = tf.nn.tanh(tf.matmul(a3_drop, W4) + b4) # output
W5 = weight_variable([25, OUTPUT_NUM]) # weight
b5 = bias_variable([OUTPUT_NUM]) # bias
a4_drop = tf.nn.dropout(a4, keep_prob)

y = tf.nn.softmax(tf.matmul(a4_drop, W5) + b5) # output



# Define optimizer
t = tf.placeholder(tf.float32, [None, OUTPUT_NUM]) # target
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y+1e-8), reduction_indices = [1]))
L2_sqr = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)+ tf.nn.l2_loss(W4)+ tf.nn.l2_loss(W5)
lambda_2 = 0.01
loss = cross_entropy + lambda_2 * L2_sqr
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#学習率が大きくてlossが上がってしまうのでは？


#tf.summary.scalar




# Add summary for data aquisition
w1_hist = tf.summary.histogram("weights", W1)
b1_hist = tf.summary.histogram("biases", b1)
w2_hist = tf.summary.histogram("weights", W2)
b2_hist = tf.summary.histogram("biases", b2)

t_hist = tf.summary.histogram("y", y)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
correct_prediction_1 = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32))




#tensorboard
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./study", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('accuracy_1', accuracy_1)
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "/home/pi/powerrecog_su/powerrecog01/Parameter0117abssingle")


#learning







spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 2000000

def pause_plot():

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)
    ax3 = fig.add_subplot(111)
    ax4 = fig.add_subplot(111)
    ax5 = fig.add_subplot(111)
    ax6 = fig.add_subplot(111)

    x1 = np.arange(-0.3, 1.3, 0.1)
    x2 = np.arange(-0.3, 1.3, 0.1)
    x3 = np.arange(-0.3, 1.3, 0.1)
    x4 = np.arange(-0.3, 1.3, 0.1)
    x5 = np.arange(-0.3, 1.3, 0.1)
    x6 = np.arange(-0.3, 1.3, 0.1)

    y1 = np.arange(-0.3, 1.3, 0.1)
    y2 = np.arange(-0.3, 1.3, 0.1)
    y3 = np.arange(-0.3, 1.3, 0.1)
    y4 = np.arange(-0.3, 1.3, 0.1)
    y5 = np.arange(-0.3, 1.3, 0.1)
    y6 = np.arange(-0.3, 1.3, 0.1)


    plt.xlabel("time[s]")
    plt.ylabel("accuracy")

    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')

    lines1, = ax1.plot(x1, y1, linewidth = 3.0, label="OFF")
    lines2, = ax2.plot(x2, y2, linewidth = 3.0, label="fan")
    lines3, = ax3.plot(x3, y3, linewidth = 3.0, label="LED")
    lines4, = ax4.plot(x4, y4, linewidth = 3.0, label="dryer")
    lines5, = ax5.plot(x5, y5, linewidth = 3.0, label="heater")
    lines6, = ax6.plot(x6, y6, linewidth = 3.0, label="microwave")
    plt.legend() # 凡例を表示


    while 1:
        while True:
            data1, data2 = readAdc()
            if data1 == 0:
                break
        while True:
            data1, data2 = readAdc()
            if data1 >= 5:
                prevtime = time.perf_counter()
                data1, data2 = readAdc()
                break

        testdataset = np.empty((0,INPUT_NUM), np.float32)

        data = []

        for i in range(SAMPLES*REC_NUM):
            while prevtime + interval > time.perf_counter():
                pass
            prevtime += interval
            data1,data2= readAdc()
            data.append(data2)

        dataarray = np.array(data, dtype=np.float32) - (2**11)
        t = np.abs(np.fft.fft( dataarray[0:SAMPLES*REC_NUM].reshape([REC_NUM,SAMPLES]) )[:,0:INPUT_NUM]).astype(np.float32)
        testdataset = np.append(testdataset, t, axis = 0)



        result = sess.run(y, feed_dict={x: testdataset, keep_prob: 1.0})


        x1 += 0.1
        x2 += 0.1
        x3 += 0.1
        x4 += 0.1
        x5 += 0.1
        x6 += 0.1

        y1 = result[0,5]
        y2 = result[0,2]
        y3 = result[0,1]
        y4 = result[0,0]
        y5 = result[0,3]
        y6 = result[0,4]


        lines1.set_data(x1, y1)
        lines2.set_data(x2, y2)
        lines3.set_data(x3, y3)
        lines4.set_data(x4, y4)
        lines5.set_data(x5, y5)
        lines6.set_data(x6, y6)


        ax1.set_xlim((x1.min(), x1.max()))
        ax2.set_xlim((x2.min(), x2.max()))
        ax3.set_xlim((x3.min(), x3.max()))
        ax4.set_xlim((x4.min(), x4.max()))
        ax5.set_xlim((x5.min(), x5.max()))
        ax6.set_xlim((x6.min(), x6.max()))


        plt.title("%s: %5.3f" % (LABELS[np.argmax(result[0,:])],np.max(result[0,:])))

        plt.pause(.002)

if __name__ == "__main__":
    pause_plot()



f.close()
spi.close()
sys.exit(0)

sess.close()
