import torch
import torch.nn.functional as F


def get_lr_schedule(nproc, warmup_epochs, milestone, decay=0.1):
	def lr_schedule(epoch):
		scale = 1
		if epoch < warmup_epochs:
			scale = 1 / nproc * (epoch * (nproc - 1) / warmup_epochs + 1)
		else:
			milestone.sort(reverse=True)
			for e in milestone:
				if epoch >= e:
					scale *= decay
		return scale
	return lr_schedule
