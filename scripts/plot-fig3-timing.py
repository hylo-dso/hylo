import matplotlib.pylab as plt
import numpy as np
import os.path

data_hylo = {}
data_sngd = {}

names_hylo = ['hylo-timing-r0-n8', 'hylo-timing-r0-n16', 'hylo-timing-r0-n32', 'hylo-timing-r0-n64']
names_sngd = ['sngd-timing-r0-n8', 'sngd-timing-r0-n16', 'sngd-timing-r0-n32', 'sngd-timing-r0-n64']
ext = '.csv'

def read_file_(names, ext):
  data = {}
  min_num_lines = float('inf')
  for name in names:
    if os.path.exists(name+ext):

      data[name] = []
      f = open(name+ext)
      f.readline()
      num_lines = 0
      for x in f:
        y = list(map(float, (x.strip().split(','))))
        data[name].append(y)
        num_lines += 1
      f.close()
      min_num_lines = min(min_num_lines, num_lines)

  for name in names:
      data[name] = data[name][:min_num_lines]
  return data

data_hylo = read_file_(names_hylo, ext)
data_sngd = read_file_(names_sngd, ext)




def get_times_(data):
  comm = []
  comp = []    
  Comp_Time = {}
  Comm_Time = {}
  for key in data:
    Comp_Time[key] = []
    Comm_Time[key] = []
    for d in data[key]:
      Comm_Time[key].append(d[0] + d[4])
      Comp_Time[key].append(d[1] + d[2] + d[3] + d[7])
    comm.append(np.median(Comm_Time[key]))
    comp.append(np.median(Comp_Time[key]))
  return comm, comp

HyLo_comm, HyLo_comp = get_times_(data_hylo)
SNGD_comm, SNGD_comp = get_times_(data_sngd)
ind = np.arange(4)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind, HyLo_comp, color='b', width = 0.25)
ax.bar(ind, HyLo_comm, bottom=HyLo_comp, color='darkblue', width = 0.25)
labels = ['', '8', '', '16', '', '32', '','64']
ax.set_xticklabels(labels)
ax.bar(ind + 0.27, SNGD_comp, color='red', width = 0.25)
ax.bar(ind + 0.27, SNGD_comm, bottom=SNGD_comp, color='darkred',  width = 0.25)
ax.set_ylabel('time(ms)')
ax.set_xlabel('GPU Count')

ax.legend(labels=['HyLo Computation', 'HyLo Communication', 'SNGD Computation', 'SNGD Communication'])
plt.show()

fig.savefig('timing.png', bbox_inches='tight')
