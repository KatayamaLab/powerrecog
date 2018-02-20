import numpy as np
import matplotlib.pyplot as plt

accgraph = np.array([])

for i in range(100):
  accgraph = np.append(accgraph,i)
  if i%10==0:
    accgraph = np.append(accgraph,i+1)

print(accgraph)

plt.plot(accgraph)
plt.show()
