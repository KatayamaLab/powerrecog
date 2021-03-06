import numpy as np
import matplotlib.pyplot as plt

accgraph_1 = np.load('realimag_originaru_acc0112.npy')

plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.plot(accgraph_1)

plt.legend() # 凡例を表示

plt.title("accurcy")
plt.xlabel("epochs[times]")
plt.ylabel("accuracy[%]")




filename = "realimag_accuracy_0115.png"
plt.savefig(filename)
