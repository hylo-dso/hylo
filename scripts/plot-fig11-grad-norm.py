import matplotlib.pylab as plt
import numpy as np
import os

nmodules = 32 # for resnet32
data = [[] for _ in range(nmodules)]
name = 'grad-norm.csv'
nsteps = 0

if os.path.exists(name):
    f = open(name)
    f.readline()
    for x in f:
        y = list(map(float, (x.strip().split(',')[:-1])))
        for m in range(len(y)):
            data[m].append(y[m])
        nsteps += 1
    f.close()

# lines to plot
plt.plot(data[4], color='skyblue', linewidth=5)
plt.plot(data[11], color='royalblue', linewidth=5)
plt.plot(data[18], color='dodgerblue', linewidth=5)
plt.plot(data[20], color='mediumslateblue', linewidth=5)
plt.plot(data[31], color='blue', linewidth=5)

print('data:', data)

# highlight critical regimes
for i in range(14, min(35, nsteps)):
    for m in range(nmodules):
        data[m][i] = None

for i in range(40, min(75, nsteps)):
    for m in range(nmodules):
        data[m][i] = None

for i in range(80, nsteps):
    for m in range(nmodules):
        data[m][i] = None

plt.plot(data[4][:nsteps], color='orange', linewidth=5, alpha=0.8)
plt.plot(data[11][:nsteps], color='orange', linewidth=5, alpha=0.8)
plt.plot(data[18][:nsteps], color='orange', linewidth=5, alpha=0.8)
plt.plot(data[20][:nsteps], color='orange', linewidth=5, alpha=0.8)
plt.plot(data[31][:nsteps], color='orange', linewidth=5, alpha=1)

plt.ylim(top=1.2, bottom=0)
plt.xlim(left=-1, right=98)
plt.ylabel('gradient norm')
plt.xlabel('epoch')
plt.legend(['Layer 4', 'Layer 11', 'Layer 18', 'Layer 20', 'Layer 31'])

plt.axvline(35, color='cadetblue', linestyle='--', linewidth=2)
plt.axvline(75, color='cadetblue', linestyle='--', linewidth=2)

plt.show()
plt.savefig('grad-norm.jpg', bbox_inches='tight')

