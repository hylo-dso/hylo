import matplotlib.pylab as plt
import numpy as np
import os.path

data = {}
names = ['grad-error-resnet50-cifar10-kid.txt', 'grad-error-resnet50-cifar10-kis.txt']

for name in names:
  if os.path.exists(name):
    f = open(name)
    data[name] = []
    for x in f:
        data[name].append(float(x.strip()))
    f.close()


plt.plot(data[names[0]], color = 'red')
plt.plot(data[names[1]], color = 'blue')

plt.xlabel('iteration')
plt.ylabel('normalized gradient error(%)')
plt.legend(['KID', 'KIS'])
plt.savefig('grad-error-resnet50-imagenet.png', bbox_inches='tight')
plt.show()
