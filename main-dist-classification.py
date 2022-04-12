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


# settings
parser = argparse.ArgumentParser(description='Distributed Second-order Optimizer')
parser.add_argument('--data-dir', type=str, default='./datasets')
parser.add_argument('--log-dir', type=str, default='./logs')
parser.add_argument('--model', type=str, default='resnet32')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.3)
parser.add_argument('--lr-decay', type=float, default=0.35)
parser.add_argument('--milestone', nargs='+', type=int, default=[30,50,70,90])
parser.add_argument('--warmup-epochs', type=int, default=5)
parser.add_argument('--warmup-init-lr', type=int, default=0.1)
parser.add_argument('--damping', type=float, default=1)
parser.add_argument('--target-damping', type=float, default=1)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--weight-decay', type=float, default=0.0004)
parser.add_argument('--freq', type=int, default=10)
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
    train_acc = Metric('train_acc')

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
                train_acc.update(backend, accuracy(outputs, target))

            if optimizer.steps % optimizer.freq == 0:
                optimizer.acc_stats = True
            loss.backward()
            optimizer.acc_stats = False

            optimizer.step()

            t.set_postfix_str("loss: {:.4f}, acc: {:.2f}%, lr: {:.4f} damp: {:.4f}".format(
                train_loss.avg, 100*train_acc.avg,
                optimizer.param_groups[0]['lr'],
                optimizer.damping))
            t.update(1)

    return train_loss, train_acc


def test(epoch, model, loss_fn, test_loader, backend, verbose):
    model.eval()
    test_loss = Metric('test_loss')
    test_acc = Metric('test_acc')

    with tqdm(total=len(test_loader),
        bar_format='{l_bar}{bar:10}|{postfix}',
        desc='             '.format(epoch, args.epochs),
        disable=not verbose) as t:
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss.update(backend, loss_fn(output, target))
                test_acc.update(backend, accuracy(output, target))

                t.update(1)
                if idx + 1 == len(test_loader):
                    t.set_postfix_str("\b\b test loss: {:.4f}, test acc: {:.2f}%".format(
                        test_loss.avg, 100*test_acc.avg), refresh=False)

    return test_loss, test_acc


def log(epoch, loss, acc, log_writter, optimizer=None, train=False):
    if train:
        log_writter.add_scalar('train/loss', loss.avg, epoch)
        log_writter.add_scalar('train/accuracy', acc.avg, epoch)
        log_writter.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    else:
        log_writter.add_scalar('test/loss', loss.avg, epoch)
        log_writter.add_scalar('test/accuracy', acc.avg, epoch)


def save_checkpoint(model, optimizer, scheduler, filepath):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    torch.save(state, filepath)


def main():
    if not torch.cuda.is_available():
        print('Error: CUDA is not available.')
        raise RuntimeError

    device = 'cuda'
    args.local_rank = int(os.environ['LOCAL_RANK'])
    print('Start init process group...')
    torch.distributed.init_process_group(
        backend=args.backend,
        init_method=args.url,
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK']))
    print('done')
    backend = get_comm_backend()

    torch.cuda.set_device(args.local_rank)
    torch.cuda.manual_seed(0)

    print('rank = {}, world size = {}, device ids = {}'.format(
        backend.rank(), backend.size(), args.local_rank))

    train_sampler, train_loader, test_sampler, test_loader = get_dataset(
        dataset=args.dataset, data_dir=args.data_dir, batch_size=args.batch_size,
        world_size=backend.size(), rank=backend.rank(), local_rank=args.local_rank)

    model = get_model(name=args.model)
    model.to(device)

    verbose = True if dist.get_rank() == 0 else False
    if verbose:
        if args.dataset == 'cifar10':
            summary(model, (args.batch_size, 3, 32, 32), device=device)
        elif args.dataset == 'imagenet':
            summary(model, (args.batch_size, 3, 224, 224), device=device)
        else:
            raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank])

    os.makedirs(args.log_dir, exist_ok=True)
    log_writter = SummaryWriter(args.log_dir) if verbose else None

    args.checkpoint_format = 'checkpoint_{epoch}.pth.tar'
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)
    args.resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break
    
    optimizer = get_optimizer(
        model, args.lr, args.damping, args.target_damping, args.weight_decay, args.freq, args.momentum, 
        args.kl_clip, args.batch_size, len(train_loader), args.warmup_epochs, backend, args.compression_ratio,
        args.profiling, args.grad_norm, args.rank_analysis, args.adaptive)

    lr_schedule = get_lr_schedule(
        backend.size(), args.warmup_epochs, args.milestone, args.lr_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    if args.dataset == 'cifar10':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif args.dataset == 'imagenet':
        loss_fn = LabelSmoothLoss(args.label_smoothing)
    else:
        raise NotImplementedError

    if args.resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        map_location = {'cuda:0': 'cuda:{}'.format(args.local_rank)}
        checkpoint = torch.load(filepath, map_location=map_location)
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if isinstance(checkpoint['scheduler'], list):
            for sched, state in zip(lr_schedule, checkpoint['scheduler']):
                sched.load_state_dict(state)

    acc = []
    timing = []

    start_training = time.time()

    for epoch in range(args.resume_from_epoch + 1, args.epochs + 1):
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
        
        test_loss, test_acc = test(epoch, model, loss_fn, test_loader, backend, verbose)

        acc.append(test_acc.avg)

        if verbose:
            log(epoch, test_loss, test_acc, log_writter)

        lr_scheduler.step()

        if epoch > 0 and epoch % args.checkpoint_freq == 0 and dist.get_rank() == 0:
            save_checkpoint(model.module, optimizer, lr_scheduler, args.checkpoint_format.format(epoch=epoch))

    if verbose:
        print('\nEnd-to-End time: {}'.format(datetime.timedelta(seconds=time.time() - start_training)))

    fname = args.model + '-' + args.dataset + '-lr' + str(args.lr)
    fname += '-d' + str(args.damping) + '-td' + str(args.target_damping)
    fname += '-wd' + str(args.weight_decay) + '-f' + str(args.freq)
    fname += '.csv'
    f = open(fname, 'w')
    f.write('acc, time(s), total_time(s)\n')
    total_time = 0
    for i in range(args.resume_from_epoch, args.epochs):
        total_time += timing[i]
        line = str(acc[i]) + ', ' + str(timing[i]) + ', ' + str(total_time) + '\n'
        f.write(line)
    f.close()


if __name__ == '__main__':
    main()

