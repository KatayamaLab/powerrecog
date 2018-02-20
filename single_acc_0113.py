import numpy as np
import matplotlib.pyplot as plt

accgraph_1 = np.load('abs_makedata_acc0113.npy')


plt.plot(accgraph_1)

plt.legend() # 凡例を表示

plt.title("accurcy")
plt.xlabel("epochs")
plt.ylabel("accuracy")

filename = "makedata_abs_accuracy.png"
plt.savefig(filename)
