import numpy as np
import matplotlib.pyplot as plt

accgraph_1 = np.load('realimag_makedata_acc0114.npy')
accgraph_2 = np.load('abs_makedata_acc0114.npy')

plt.plot(accgraph_1, color="blue", alpha=0.8, label="real imaginary part")
plt.plot(accgraph_2, color="red", alpha=0.8, label="absolute value")
plt.legend() # 凡例を表示

plt.title("accurcy")
plt.xlabel("epochs")
plt.ylabel("accuracy")

filename = "compete_makedata_accuracy.png"
plt.savefig(filename)
