# Fig 7
import matplotlib.pylab as plt
import numpy as np
import os.path

data_hylo = {}
data_sngd = {}

names_resnet50 = ['./img/kid-timing-r0-n64', './img/kis-timing-r0-n64']
names_resnet32 = ['./cifar10/kid-timing-r0-n32', './cifar10/kis-timing-r0-n32']
names_unet = ['./unet/kid-timing-r0-n4', './unet/kis-timing-r0-n4']
ext = '.csv'

def read_file_(names, ext):
  data = {}
  for name in names:
    if os.path.exists(name+ext):

      data[name] = []
      f = open(name+ext)
      f.readline()
      for x in f:
        y = list(map(float, (x.strip().split(','))))
        data[name].append(y)
      f.close()
  return data

data_resnet50 = read_file_(names_resnet50, ext)
data_resnet32 = read_file_(names_resnet32, ext)
data_unet = read_file_(names_unet, ext)


def get_times_(data):
  result = {}
  fac = {}
  invv = {}
  gat = {}
  brd = {}
  for key in data:
    fac[key] = []
    invv[key] = []
    gat[key] = []
    brd[key] = []
    result[key] = []
    for d in data[key]:
      fac[key].append(d[1])
      invv[key].append(d[3])
      gat[key].append(d[0])
      brd[key].append(d[4])
    result[key] = [np.median(fac[key]), np.median(invv[key])
    ,np.median(gat[key]), np.median(brd[key])]
  return result

result_resnet50 = get_times_(data_resnet50)
result_resnet32 = get_times_(data_resnet32)
result_unet = get_times_(data_unet)

def plot_timing(result, fname):
  ind = np.arange(4)
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  colors = ['r', '#FF7F50']
  i = 0
  for key in result:
    ax.bar(ind + 0.27 * i, result[key], color=colors[i], width = 0.25)
    i += 1
  labels = ['', 'Factorization', '', 'Inversion', '', 'Gather', '','Broadcast']
  ax.set_xticklabels(labels)
  ax.set_ylabel('time(ms)')
  ax.legend(labels=['HyLo-KID', 'HyLo-KIS'])
  plt.show()
  fig.savefig(fname, bbox_inches='tight')

plot_timing(result_resnet50, 'fig7-resnet50.jpg')
plot_timing(result_resnet32, 'fig7-resnet32.jpg')
plot_timing(result_unet, 'fig7-unet.jpg')
