import argparse
import datetime
# import dso
import os
import sys
import time
import torch
import torch.distributed as dist

from dso.utils.comm import get_comm_backend
from dso.utils.datasets import get_dataset
from dso.utils.metric import accuracy, Metric, LabelSmoothLoss
from dso.utils.models import get_model
from dso.utils.optimizers import get_optimizer
from dso.utils.scheduler import get_lr_schedule

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# settings
parser = argparse.ArgumentParser(description='HyLo: Hybrid and Low-Rank Natural Gradient Descent')
parser.add_argument('--data-dir', type=str, default='./datasets', help='path to the directory where the dataset is stored/will be downloaded to')
parser.add_argument('--log-dir', type=str, default='./logs', help='path to the directory where the training log will be written to')
parser.add_argument('--model', type=str, default='resnet32', help='name of the neural network')
parser.add_argument('--dataset', type=str, default='cifar10', help='name of the dataset')
parser.add_argument('--batch-size', type=int, default=128, help='local batch size, i.e. batch size per GPU/worker')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.3, help='standard learning rate (before decay and after warm up)')
parser.add_argument('--lr-decay', type=float, default=0.35, help='learning rate decay rate')
parser.add_argument('--milestone', nargs='+', type=int, default=[30,50,70,90], help='epoch(s) at which the learning rate is decayed')
parser.add_argument('--warmup-epochs', type=int, default=5, help='number of epochs to warm up the learning rate for distributed training')
parser.add_argument('--damping', type=float, default=1, help='(initial) damping parameter to stablize inversion in the optimizer')
parser.add_argument('--target-damping', type=float, default=1, help='target damping value in the damping decay schedule')
parser.add_argument('--damping-decay-epochs', type=int, default=90, help='number of steps to reach target damping from the initial damping value')
parser.add_argument('--momentum', type=float, default=0.90, help='momentum')
parser.add_argument('--weight-decay', type=float, default=0.0004, help='weight decay')
parser.add_argument('--freq', type=int, default=10, help='inverse frequency in the optimizer, to update preconditioning matrices')
parser.add_argument('--compression-ratio', type=float, default=0.1, help='compression ratio of HyLo')
parser.add_argument('--label-smoothing', type=float, default=0.1, help='parameter in the loss function for imagenet')
parser.add_argument('--checkpoint-freq', type=int, default=5, help='frequency at which the checkpoint is saved')
parser.add_argument('--backend', type=str, default='nccl', help='only NCCL backend is supported right now')
parser.add_argument('--profiling', action='store_true', help='enable profiling of the optimizer')
parser.add_argument('--grad-norm', action='store_true', help='enable gradient norm analysis of the optimizer')
parser.add_argument('--grad-error', action='store_true', help='enable gradient error analysis of KID/KIS')
parser.add_argument('--rank-analysis', action='store_true', help='enable rank analysis of the optimizer')
parser.add_argument('--url', type=str, default='env://', help='url used to initialize the process group')
parser.add_argument('--local_rank', type=int, default=0, help='local rank, to be set automatically by torch.distributed.launch')
parser.add_argument('--adaptive', nargs='+', type=int, default=None, help='epoch(s) at which HyLo performs switching between KID/KIS')
parser.add_argument('--enable-is', action='store_true', help='enable KIS, only use KIS throughout the training')
parser.add_argument('--enable-id', action='store_true', help='enable KID, only use KID throughout the training')
parser.add_argument('--sngd', action='store_true', help='enable SNGD, only use SNGD throughout the training')

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

    randk = False
    if args.enable_id:
        randk = False
    if args.enable_is:
        randk = True
    if args.enable_id and args.enable_is:
        raise ValueError("One and only one of KIS and KIS should be enabled.")
    '''
    if not args.enable_id and not args.enable_is:
        raise ValueError("One and only one of KIS and KIS should be enabled.")
    '''
    prefix = ''
    if args.rank_analysis:
        prefix = args.dataset + '-' + str(args.batch_size) + '-'
    optimizer = get_optimizer(
        model, args.lr, args.damping, args.target_damping, args.damping_decay_epochs, args.weight_decay, args.freq, args.momentum, 
        args.batch_size, len(train_loader), args.warmup_epochs, backend, randk, args.compression_ratio,
        args.profiling, args.grad_norm, args.grad_error, args.rank_analysis, args.adaptive, args.sngd,
        args.enable_id, args.enable_is, prefix)

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

        acc.append(test_acc.avg.item())

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
    f.write('acc, time(ms), total_time(ms)\n')
    total_time = 0
    for i in range(args.resume_from_epoch, args.epochs):
        total_time += timing[i]
        line = str(acc[i]) + ', ' + str(timing[i]) + ', ' + str(total_time) + '\n'
        f.write(line)
    f.close()

    if args.grad_error and backend.rank() == 0:
        fname = 'grad-error-' + args.model + '-' + args.dataset
        if args.enable_id:
            fname += '-kid.txt'
        elif args.enable_is:
            fname += '-kis.txt'

        f = open(fname, "w")
        for k in optimizer.grad_error:
            f.write(str(k) + '\n')
        f.close()


if __name__ == '__main__':
    main()

