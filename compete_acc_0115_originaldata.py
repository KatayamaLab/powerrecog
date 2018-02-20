import numpy as np
import matplotlib.pyplot as plt

accgraph_1 = np.load('realimag_originaru_acc0112.npy')
accgraph_2 = np.load('abs_originaru_acc0112.npy')


plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')

plt.plot(accgraph_1, color="blue", alpha=0.8, label="real imaginary part")
plt.plot(accgraph_2, color="red", alpha=0.8, label="absolute value")
plt.legend() # 凡例を表示

plt.title("accurcy")
plt.xlabel("epochs[times]")
plt.ylabel("accuracy[%]")
plt.ylim([0,100])

filename = "compete_originaldata_accuracy_0124.png"
plt.savefig(filename)
