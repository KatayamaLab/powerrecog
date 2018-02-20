import numpy as np

a = ['あ','い','う','え','お']
b = np.arange(6)
print(b)
c = b.reshape(1,6)
print(c[0,1])

for ai,bi in zip(a, c[0,:]):
  print(ai,bi)
