import argparse
import datetime
import dso
import os
import sys
import time
import torch
import torch.distributed as dist
from tqdm import tqdm

from dso.optimizers import NGDOptimizer

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from dso.utils.comm import get_comm_backend
from dso.utils.datasets import get_dataset
from dso.utils.metric import accuracy, Metric, LabelSmoothLoss
from dso.utils.models import get_model
from dso.utils.optimizers import get_optimizer
from dso.utils.scheduler import get_lr_schedule
import numpy as np
from dso.utils.cnn_utils.unet import DiceLoss
from medpy.filter.binary import largest_connected_component
import math

# settings
parser = argparse.ArgumentParser(description='Distributed Second-order Optimizer')
parser.add_argument('--data-dir', type=str, default='./datasets')
parser.add_argument('--log-dir', type=str, default='./logs')
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--dataset', type=str, default='brain-segmentation')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0008)
parser.add_argument('--lr-decay', type=float, default=1)
parser.add_argument('--milestone', nargs='+', type=int, default=[100])
parser.add_argument('--warmup-epochs', type=int, default=10)
parser.add_argument('--damping', type=float, default=0.03)
parser.add_argument('--target-damping', type=float, default=0.03)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--weight-decay', type=float, default=0.0001)
parser.add_argument('--freq', type=int, default=100)
parser.add_argument('--kl-clip', type=int, default=0.01)
parser.add_argument('--compression-ratio', type=float, default=0.1)
parser.add_argument('--label-smoothing', type=float, default=0.1)
parser.add_argument('--checkpoint-freq', type=int, default=5)
parser.add_argument('--backend', type=str, default='nccl')
parser.add_argument('--profiling', action='store_true')
parser.add_argument('--grad-norm', action='store_true')
parser.add_argument('--rank-analysis', action='store_true')
parser.add_argument('--url', type=str, default='env://')
parser.add_argument('--node-idx', type=int, default=0)
parser.add_argument('--nproc-per-node', type=int, default=4)
parser.add_argument('--world-size', type=int, default=64)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--adaptive', nargs='+', type=int, default=None)

args = parser.parse_args()


def train(epoch, model, optimizer, loss_fn, batch_size, train_sampler, train_loader, backend, verbose):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')

    with tqdm(total=len(train_loader),
        bar_format='{l_bar}{bar:10}{r_bar}',
        desc='Epoch {:3d}/{:3d}'.format(epoch, args.epochs),
        disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            outputs = model(data)
            loss = loss_fn(outputs, target)

            with torch.no_grad():
                train_loss.update(backend, loss)


            if optimizer.steps % optimizer.freq == 0:
                optimizer.acc_stats = True
            loss.backward()
            optimizer.acc_stats = False

            optimizer.step()

            t.set_postfix_str("loss: {:.4f}, lr: {:.4f}".format(
                train_loss.avg,
                optimizer.param_groups[0]['lr']))
            t.update(1)

    return train_loss, None


def test(epoch, model, loss_fn, test_loader, backend, verbose):
    model.eval()
    test_loss = Metric('test_loss')

    with tqdm(total=len(test_loader),
        bar_format='{l_bar}{bar:10}|{postfix}',
        desc='             '.format(epoch, args.epochs),
        disable=not verbose) as t:
        validation_pred, validation_true = [], []
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss =  loss_fn(output, target)
                test_loss.single_thread_update(loss)

                output_np, target_np = output.detach().cpu().numpy(), target.detach().cpu().numpy()
                validation_pred.extend([output_np[s] for s in range(output_np.shape[0])])
                validation_true.extend([target_np[s] for s in range(target_np.shape[0])])

                t.update(1)
                if idx + 1 == len(test_loader):
                    mean_dsc = np.mean(dsc_per_volume(validation_pred, validation_true, test_loader.dataset.patient_slice_index,))
                    if backend.rank() == 0: print("epoch {}, val_loss: {:.4f}, val_mean_dsc_value: {:.4f}".format(epoch, test_loss.avg, mean_dsc))

    return test_loss, mean_dsc


def dsc(y_pred, y_true, lcc=True):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    if lcc and np.any(y_pred): 
        y_pred = largest_connected_component(y_pred)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list


def log(epoch, loss, acc, log_writter, optimizer=None, train=False):
    if train:
        log_writter.add_scalar('train/loss', loss.avg, epoch)
        log_writter.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    else:
        log_writter.add_scalar('test/loss', loss.avg, epoch)
        log_writter.add_scalar('test/accuracy', acc, epoch)


def save_checkpoint(model, optimizer, scheduler, filepath):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    torch.save(state, filepath)


def main():
    #torch.multiprocessing.set_start_method('spawn')
    if not torch.cuda.is_available():
        print('Error: CUDA is not available.')
        raise RuntimeError
    device = 'cuda'
    args.local_rank = int(os.environ['LOCAL_RANK'])
    print('Start init process group...')
    torch.distributed.init_process_group(backend=args.backend, init_method=args.url,
        world_size=args.world_size, rank=(args.node_idx * args.nproc_per_node)+args.local_rank)
    print('done')
    backend = get_comm_backend()

    torch.cuda.set_device(args.local_rank)

    print('rank = {}, world size = {}, device ids = {}'.format(
        backend.rank(), backend.size(), args.local_rank))

    train_sampler, train_loader, test_sampler, test_loader = get_dataset(
        dataset=args.dataset, data_dir=args.data_dir, batch_size=args.batch_size,
        world_size=backend.size(), rank=backend.rank(), local_rank=args.local_rank)

    model = get_model(name=args.model)
    model.to(device)

    verbose = True if dist.get_rank() == 0 else False
    if verbose:
        if args.dataset == 'brain-segmentation':
            summary(model, (args.batch_size, 3, 256, 256), device=device)
        else:
            raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank])
    dir_name = args.log_dir + '/exp_lr'+ str(args.lr) + '_wd' + str(args.weight_decay) + '_d' + str(args.damping)
    os.makedirs(dir_name, exist_ok=True)
    log_writter = SummaryWriter(dir_name) if verbose else None

    optimizer = get_optimizer(
        model, args.lr, args.damping, args.target_damping, args.weight_decay, args.freq, args.momentum,
        args.kl_clip, args.batch_size, len(train_loader), args.warmup_epochs, backend, args.compression_ratio,
        args.profiling, args.grad_norm, args.rank_analysis, args.adaptive)

    lr_schedule = get_lr_schedule(
        backend.size(), args.warmup_epochs, args.milestone, args.lr_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    if args.dataset == 'brain-segmentation':
        loss_fn = DiceLoss()
    else:
        raise NotImplementedError

    acc = []
    timing = []

    start_training = time.time()

    for epoch in range(1, args.epochs + 1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        train_loss, train_acc = train(epoch, model, optimizer, loss_fn, args.batch_size,
            train_sampler, train_loader, backend, verbose)

        end.record()
        torch.cuda.synchronize()
        timing.append(start.elapsed_time(end))

        if verbose:
            log(epoch, train_loss, train_acc, log_writter, optimizer, train=True)
        
        test_loss, mean_dsc = test(epoch, model, loss_fn, test_loader, backend, verbose)

        acc.append(mean_dsc)

        if verbose:
            log(epoch, test_loss, mean_dsc, log_writter)

        lr_scheduler.step()

        if epoch > 0 and epoch % args.checkpoint_freq == 0 and dist.get_rank() == 0:
            save_checkpoint(model.module, optimizer, lr_scheduler,
                args.log_dir + '/' + 'checkpoint_{}.pth.tar'.format(epoch))

    if verbose:
        print('\nTraining time: {}'.format(datetime.timedelta(seconds=time.time() - start_training)))

    fname = args.model + '-' + args.dataset + '-lr' + str(args.lr)
    fname += '-d' + str(args.damping) + '-td' + str(args.target_damping)
    fname += '-wd' + str(args.weight_decay) + '-f' + str(args.freq)
    fname += '.csv'
    f = open(fname, 'w')
    f.write('acc, time(s), total_time(s)\n')
    total_time = 0
    for i in range(0, args.epochs):
        total_time += timing[i]
        line = str(acc[i]) + ', ' + str(timing[i]) + ', ' + str(total_time) + '\n'
        f.write(line)
    f.close()


if __name__ == '__main__':
    main()

