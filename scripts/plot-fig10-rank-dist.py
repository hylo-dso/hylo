import matplotlib.pylab as plt
import numpy as np
import os.path


names_cifar = ['./cifar10-512-rank', './cifar10-1024-rank', './cifar10-2048-rank', './cifar10-4096-rank']
names_img = ['./img-512-rank', './img-1024-rank', './img-2048-rank', './img-4096-rank']
ext = '.csv'

def read_file_(names):
  r = {}
  for filename in names:
    r[filename] = []
    for i in range(0, 64):
        name = filename + '-' + str(i)
        if os.path.exists(name+ext):
            f = open(name+ext)
            f.readline()
            for x in f:
                results = x.strip().split(',')
                r[filename] += list(map(int, results[:-1]))
            f.close()
  return r

def plot_rank(ranks, fname):

  data = []
  for k in ranks:
    data.append(ranks[k])

  fig = plt.figure(figsize=(10, 7))
  
  # Creating axes instance
  ax = fig.add_axes([0, 0, 1, 1])
  
  # Creating plot
  labels = [512, 1024, 2048, 4096]

  bp = ax.boxplot(data)
  
  ax.set_yscale('log') 
  ax.set_xlabel('global batch size')
  ax.set_ylabel('rank')
  ax.set_xticklabels(labels)
  # plt.plot(labels, labels)
  plt.show()
  fig.savefig(fname, bbox_inches='tight')

rank_img = read_file_(names_img)
rank_cifar = read_file_(names_cifar)
plot_rank(rank_cifar, 'rank-cifar-resnet32.jpg')
plot_rank(rank_img, 'rank-imagenet-resnet50.jpg')



  
