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

    initial =  tf.truncated_normal(shape, stddev=0.14)
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
L2_sqr = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)+ tf.nn.l2_loss(W3)
lambda_2 = 0.01
loss = cross_entropy + lambda_2 * L2_sqr
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
#学習率が大きくてlossが上がってしまうのでは？

#tf.summary.scalar




# Add summary for data aquisition
w1_hist = tf.summary.histogram("weights", W1)
b1_hist = tf.summary.histogram("biases", b1)
w2_hist = tf.summary.histogram("weights", W2)
b2_hist = tf.summary.histogram("biases", b2)
w3_hist = tf.summary.histogram("weights", W3)
b3_hist = tf.summary.histogram("biases", b3)
t_hist = tf.summary.histogram("y", y)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
correct_prediction_1 = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32))



sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./study", sess.graph)



init = tf.global_variables_initializer()
sess.run(init)



tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('accuracy_1', accuracy_1)
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

for i in range(1000000):
  if i%100==0:
    summary, acc = sess.run([merged,accuracy], feed_dict={x: testdataset, t: testydata, keep_prob: 1.0})
    writer.add_summary(summary, i)
    acc_1 = sess.run(accuracy, feed_dict={x: dataset, t: ydata, keep_prob: 1.0})
    print("step = %s acc = %s acc_1 = %s" % (i, acc, acc_1) )
  sess.run(train_step, feed_dict={x: dataset, t: ydata, keep_prob: 0.5})




#    GPIO.output(port,True)
#    GPIO.output(port,False)
#            time.sleep(0.01)







sys.exit(0)

sess.close()
