import matplotlib.pylab as plt
import numpy as np
import os.path

acc = []
time = []
epochs = 0

name = 'resnet32-cifar10-lr1.8-d1.5-td0.1-wd0.00045-f13.csv'

if os.path.exists(name):
    f = open(name)
    f.readline()
    for x in f:
        y = list(map(float, (x.strip().split(','))))
        acc.append(y[0])
        time.append(y[2])
        epochs += 1

plt.plot(acc)
plt.title('ResNet-32 + CIFAR-10')
plt.ylabel('test accuracy')
plt.xlabel('epoch')
plt.legend(['HyLo'])
plt.xlim(right=100)
plt.show()
plt.savefig('acc-vs-epoch-resnet32-cifar10.jpg', bbox_inches='tight')
plt.cla()

print('time:', time)
print('acc:', acc)
plt.plot(time, acc)
plt.title('ResNet-32 + CIFAR-10')
plt.ylabel('test accuracy')
plt.xlabel('wall-clock time')
plt.legend('HyLo')
plt.xlim(right=100)
plt.show()
plt.savefig('acc-vs-time-resnet32-cifar10.jpg', bbox_inches='tight')

