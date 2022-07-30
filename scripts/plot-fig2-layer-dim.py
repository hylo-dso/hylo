import torch

from torchsummary import summary

import math
import time
import copy
import matplotlib.pylab as plt
import numpy as np
from dso.utils.models import get_model
import torchvision.models as models

dd = {}
all_models = ['resnext50_32x4d', 'mobilenet_v3_large', 'wide_resnet50_2'
              ,'inception_v3','mnasnet1_0', 'resnet32', 'unet']

all_labels = ['ResNet-50', 'MobileNet-V3', 'Wide-ResNet'
              ,'Inception-V3','MnasNet', 'ResNet-32', 'U-Net']
labels = []
for i in range(len(all_models)):
  item = all_models[i]
  if item in ['resnet32', 'unet']:
    net = get_model(item)
    labels.append(all_labels[i])
    dd[item] = []
  else:
    try:
      net_name = getattr(models, item)
      net = net_name()
      labels.append(all_labels[i])
      dd[item] = []
    except Exception as e:
      print('Update Torch.')
      continue
    

  for m in net.modules():
    classname = m.__class__.__name__
    if classname in ['Conv2d']:
     d = m.in_channels
     k = m.kernel_size[0]
     dd[item].append(d*k*k)
    if classname in ['Linear']:
     dd[item].append(m.in_features)
   
    if classname in ['Sequential']:
    #  print(classname)
     for mm in m.modules():
       if mm.__class__.__name__ in ['Conv2d']:
          d = mm.in_channels
          k = mm.kernel_size[0]
          dd[item].append(d*k*k)
       if mm.__class__.__name__ in ['Linear']:
          dd[item].append(mm.in_features)
       
data = []
for k in dd:  
  data.append(dd[k])
 
fig = plt.figure(figsize =(10, 7))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)
ax.set_yscale('log') 
ax.set_xlabel('model')
ax.set_ylabel('layer dimension')

print(labels)
ax.set_xticklabels(labels)
# show plot
plt.show()
fig.savefig('layer-dim.jpg', bbox_inches='tight')
