import torch
import torch.nn.functional as F

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.total = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, backend, val, n=1):
        backend.allreduce(val, async_op=False)
        self.total += val.cpu()
        self.n += n

    def single_thread_update(self, val, n=1):
        self.total += val.cpu()
        self.n += n

    @property
    def avg(self):
        return self.total / self.n


def accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()



class LabelSmoothLoss(torch.nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


