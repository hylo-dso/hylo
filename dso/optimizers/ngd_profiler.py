import numpy as np
import torch

from math import ceil, floor
from torch import einsum, eye, inverse
from torch.optim import Optimizer
from ..utils.factors import ComputeI, ComputeG


class NGDProfiler(Optimizer):
    def __init__(self, model, lr, init_damping, target_damping, weight_decay, freq, momentum,
                 batch_size, iters, warmup_epochs, backend, compression_ratio=0.1,
                 profiling=False, grad_norm=False, rank_analysis=False, adaptive=None, sngd=False,
                 enable_id=False, enable_is=False, prefix=''):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=init_damping, weight_decay=weight_decay)

        super(NGDProfiler, self).__init__(model.parameters(), defaults)

        self.model = model
        self.modules = []
        self.known_modules = {'Linear', 'Conv2d'}

        self.IHandler = ComputeI()
        self.GHandler = ComputeG()

        self.m_I, self.m_II = {}, {}
        self.m_G, self.m_GG = {}, {}
        self.m_K = {}
        self.m_RDR = {}
        self.assigned_worker = {}

        self.damping = init_damping
        self.init_damping = init_damping
        self.target_damping = target_damping
        self.weight_decay = weight_decay
        self.freq = freq
        self.momentum = momentum

        self.global_bs = batch_size * backend.size()
        self.iters_per_epoch = iters
        self.damping_decay = self.iters_per_epoch * warmup_epochs

        self.backend = backend

        self.compression_ratio = compression_ratio
        self.profiling = profiling
        self.grad_norm = grad_norm
        self.rank_analysis = rank_analysis

        self.ranks = []

        self.adaptive = adaptive
        self.randk = False

        self.enable_id = enable_id
        self.enable_is = enable_is
        if enable_is:
            self.randk = True
            print('KIS is enabled throughout the training.')
        elif enable_id:
            print('KID is enabled throughout the training.')

        if self.grad_norm:
            self.randk = True
            self.compression_ratio = 1
            self.adaptive = None

            fname = 'grad-norm.csv'
            f = open(fname, 'w')
            title = ''
            for i in range(len(self.modules)):
                title += str(i)
                title += ', '
            title += '\n'
            f.write(title)
            f.close()

        if self.rank_analysis:
            sngd = True
            self.prefix = prefix
            fname = prefix + 'rank-' + str(self.backend.rank()) + '.csv'
            f = open(fname, 'w')
            title = 'rank'
            title += '\n'
            f.write(title)
            f.close()


        self.timing = {}
        self.reset_timing()
        
        # sngd flag overwrite compression ratio to 1 and randk = True
        self.sngd = sngd
        if sngd:
            self.compression_ratio = 1
            self.randk = True

        if profiling:
            fname = 'hylo'
            if sngd:
                fname = 'sngd'
            elif enable_id:
                fname = 'kid'
            elif enable_is:
                fname = 'kis'
                
            fname += '-timing-r'
            fname += str(self.backend.rank())
            fname += '-n'
            fname += str(self.backend.size())
            fname += '.csv'
            
            f = open(fname, 'w')
            title = ''
            for k in self.timing:
                title += k
                title += ','
            title += '\n'
            f.write(title)
            f.close()
            
            self.fname = fname

        self.acc_stats = False
        self.steps = 0

        self._prepare_model()


    def _save_input(self, m, input):
        if torch.is_grad_enabled() and self.steps % self.freq == 0:
            self.m_I[m] = self.IHandler(input[0].data, m)


    def _save_grad_output(self, m, grad_input, grad_output):
        if self.acc_stats and self.steps % self.freq == 0:
            G, _ = self.GHandler(grad_output[0].data, m)
            self.m_G[m] = G

            

            if not self.randk:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                self.interpolative_decomposition(m)
                end.record()

                torch.cuda.synchronize()
                self.timing['id_time'] = start.elapsed_time(end)

                return

            n = G.size(0)
            
            I = self.m_I[m]
            I_norm = torch.pow(torch.norm(I, p=2, dim=1), 2)
            G_norm = torch.pow(torch.norm(G, p=2, dim=1), 2)
            norm = I_norm * G_norm
            idx = torch.multinomial(norm, num_samples=int(n * self.compression_ratio))

            self.m_G[m] = G[idx, :]
            self.m_I[m] = self.m_I[m][idx, :]


    def _assign_worker(self, m, count, avg):
        self.assigned_worker[m] = floor(count / avg)


    def _prepare_model(self):
        count = 0
        if self.backend.rank() == 0:
            print('HyLoProfiler keeps the following modules:')
        for m in self.model.modules():
            classname = m.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(m)
                m.register_forward_pre_hook(self._save_input)
                m.register_backward_hook(self._save_grad_output)
                if self.backend.rank() == 0:
                    print('(%s): %s' % (count, m))
                count += 1

        avg = ceil(count / self.backend.size())
        assigned = 0
        for m in self.modules:
            self._assign_worker(m, assigned, avg)
            assigned += 1

    def _allgather_factors(self, X):
        return [self.backend.allgather(X)]

    def allgather_factors(self):
        if self.backend.size() == 1:
            return

        Icat = []
        for m in self.modules:
            Icat.append(self.m_I[m])
            Icat.append(self.m_G[m])

        if not self.randk:
            for m in self.modules:
                Icat.append(self.m_RDR[m])
        X = torch.cat(Icat, dim=1)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        handles = []
        handles.extend(self._allgather_factors(X))
        self.backend.sync(handles)

        end.record()
        torch.cuda.synchronize()
        self.timing['allgather_factors_comm'] = start.elapsed_time(end)

        cat_time = 0
        start.record()
        _, IGR_list = handles[0]
        RDR_len = len(IGR_list)
        IGR = torch.cat(IGR_list, dim=0)
        offset = 0

        for m in self.modules:
            Isize = self.m_I[m].shape[1]
            Gsize = self.m_G[m].shape[1]
            self.m_I[m] = IGR[:, offset:offset+Isize]
            offset += Isize
            self.m_G[m] = IGR[:, offset:offset+Gsize]
            offset += Gsize

        if not self.randk:
            for m in self.modules:
                Rsize = self.m_RDR[m].shape[1]
                mat = IGR[:, offset:offset+Rsize]
                self.m_RDR[m] = [mat[i*Rsize:(i+1)*Rsize, :] for i in range(RDR_len)]
                offset += Rsize

        end.record()
        torch.cuda.synchronize()
        cat_time += start.elapsed_time(end)
        self.timing['allgather_factors_cat'] = cat_time

    @torch.no_grad()
    def _compute_II(self, m):
        I = self.m_I[m]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        II = einsum('nk,qk->nq', (I, I))

        end.record()
        torch.cuda.synchronize()
        self.timing['compute_factors'] += start.elapsed_time(end)

        return II


    @torch.no_grad()
    def _compute_GG(self, m):
        G = self.m_G[m]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        GG = einsum('nk,qk->nq', (G, G))
        end.record()
        torch.cuda.synchronize()
        self.timing['compute_factors'] += start.elapsed_time(end)

        return GG


    @torch.no_grad()
    def _interpolative_decomposition(self, K, r):
        n = K.size(0)

        Q, R = torch.qr(K)
        R11 = R[:r, :r]
        Q1 = Q[:, :r]
        Q2 = Q[:, r:]
        R11 = R[:r, :r]
        R12 = R[:r, r:]
        R22 = R[r:, r:]

        T = torch.inverse(R11) @ R12
        RT = torch.cat([torch.eye(r).to(T.device), T], dim=1)

        residual = Q2 @ R22
        residual = torch.cat([torch.zeros(n, r).to(K.device), residual], dim=1)

        return RT, residual


    @torch.no_grad()
    def interpolative_decomposition(self, m):
        I = self.m_I[m]
        G = self.m_G[m]

        # permute rows of I and G based on their norm
        I_norm = torch.pow(torch.norm(I, p=2, dim=1), 2)
        G_norm = torch.pow(torch.norm(G, p=2, dim=1), 2)
        norm = I_norm * G_norm
        _, idx = torch.sort(norm, descending=True)
        self.m_I[m] = I[idx, :]
        self.m_G[m] = G[idx, :]

        II = self._compute_II(m)
        GG = self._compute_GG(m)
        K = II * GG

        n = I.size(0)
        r = int(n * self.compression_ratio)
        RT, residual = self._interpolative_decomposition(K, r)

        assert(residual.shape[0] == residual.shape[1])
        # Dinv is m x m
        Dinv = torch.inverse(residual + self.damping * torch.eye(residual.size(0)).to(residual.device))

        # construct RDR = RT * Dinv * R: List of r x r
        self.m_RDR[m] = (RT @ Dinv @ RT.T).contiguous()
        
        self.m_I[m] = self.m_I[m][:r, :]
        self.m_G[m] = self.m_G[m][:r, :]



    @torch.no_grad()
    def compute_factors(self):
        for m in self.modules:
            if self.backend.rank() != self.assigned_worker[m]:
                continue

            self.m_II[m] = self._compute_II(m)
            self.m_GG[m] = self._compute_GG(m)


    @torch.no_grad()
    def update_inv(self):
        for m in self.modules:
            if self.backend.rank() != self.assigned_worker[m]:
                # initialize buffer for inverse kernel, if this worker
                # is not assigned to compute the inverse for this module/layer
                n = self.m_I[m].shape[0]
                self.m_K[m] = torch.empty(n, n).to(self.m_I[m].device)
                continue

            II = self.m_II[m]
            GG = self.m_GG[m]
            n = II.shape[0]

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            K = (II * GG) / n

            end.record()
            torch.cuda.synchronize()
            self.timing['compute_kernel'] += start.elapsed_time(end)

            # rank analysis
            u,s,v = torch.svd(K, compute_uv=False)
            cs = torch.cumsum(s, dim=0) / torch.sum(s)
            rank = torch.sum(cs < 0.9)
            self.ranks.append(rank.item())

            start.record()
            if self.randk:
                self.m_K[m] = torch.inverse(K + self.damping * eye(n).to(II.device))
            else:
                self.m_K[m] = torch.inverse(K)

            end.record()
            torch.cuda.synchronize()

            self.timing['update_inv'] += start.elapsed_time(end)

            self.m_II[m] = None
            self.m_GG[m] = None
            torch.cuda.empty_cache()


    def _broadcast_inv(self, m):
        if m.__class__.__name__.lower() == 'linear':
            return [self.backend.broadcast(self.m_K[m], src=self.assigned_worker[m])]
        elif m.__class__.__name__.lower() == 'conv2d':
            return [self.backend.broadcast(self.m_K[m], src=self.assigned_worker[m])]
        else:
            raise NotImplementedError


    def broadcast_inv(self):
        if self.backend.size() == 1:
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        handles = []
        for m in self.modules:
            handles.extend(self._broadcast_inv(m))
        self.backend.sync(handles)

        end.record()
        torch.cuda.synchronize()
        self.timing['broadcast_inv'] += start.elapsed_time(end)

    def compute_projection(self):
        if self.backend.size() == 1:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for m in self.modules:
                RDR = self.m_RDR[m]
                K = self.m_K[m]
                self.m_K[m] = RDR - RDR @ torch.inverse(self.m_K[m] + RDR) @ RDR

            end.record()
            torch.cuda.synchronize()
            self.timing['compute_projection'] += start.elapsed_time(end)

        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            for m in self.modules:
                RDR = torch.block_diag(*self.m_RDR[m])
                self.m_K[m] = RDR - RDR @ torch.inverse(self.m_K[m] + RDR) @ RDR
                
            end.record()
            torch.cuda.synchronize()
            self.timing['compute_projection'] += start.elapsed_time(end)

    def _precondition(self, m):
        grad = m.weight.grad.data
        bias = m.bias.grad.data if m.bias is not None else None

        classname = m.__class__.__name__.lower()

        if classname == 'linear':
            I = self.m_I[m]
            G = self.m_G[m]
            K = self.m_K[m]
            n = I.shape[0]

            if bias is not None:
                grad_reshape = torch.cat([grad, bias.view(-1, 1)], 1)

            jvp = einsum('ni,oi->no', (I, grad_reshape))
            jvp = einsum('no,no->n', (jvp, G))

            Kv = K @ jvp

            jtvp = einsum('n,no->no', (Kv, G))
            jtvp = einsum('no,ni->oi', (jtvp, I))
            jtvp = jtvp / n

            v = (grad_reshape - jtvp) / self.damping

        elif classname == 'conv2d':
            grad_reshape = grad.reshape(grad.shape[0], -1)
            I = self.m_I[m]
            G = self.m_G[m]
            K = self.m_K[m]
            n = I.shape[0]

            if bias is not None:
                grad_reshape = torch.cat([grad_reshape, bias.view(-1, 1)], 1)

            jvp = einsum('nk,mk->nm', (I, grad_reshape))
            jvp = einsum('nm,nm->n', (jvp, G))

            Kv = (K @ jvp.unsqueeze(1)).squeeze()

            jtvp = einsum('n,nm->nm', (Kv, G))
            jtvp = einsum('nm,nk->mk', (jtvp, I))
            jtvp = jtvp / n

            v = (grad_reshape - jtvp) / self.damping

        else:
            raise NotImplementedError

        if bias is not None:
            grad_update = v[:,:-1]
            bias_update = v[:,-1:]
        else:
            grad_update = v
            bias_update = None

        bias_update = bias_update.view_as(bias).contiguous() if bias_update is not None else None
        updates = grad_update.view_as(grad).contiguous(), bias_update

        return updates


    def precondition(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        updates = {}
        for m in self.modules:
            updates[m] = self._precondition(m)
        end.record()
        torch.cuda.synchronize()
        self.timing['precondition'] = start.elapsed_time(end)
        return updates


    def _broadcast_preconditioned_gradients(self, m, updates):
        return [self.backend.broadcast(tensor, src=self.assigned_worker[m]) 
                for tensor in updates[m] if tensor is not None]


    def broadcast_preconditioned_gradients(self, updates):
        if self.backend.size() == 1:
            return

        handles = []
        for m in self.modules:
            handles.extend(self._broadcast_preconditioned_gradients(m, updates))
        self.backend.sync(handles)


    def update_grad_and_step(self, updates):
        for m in self.model.modules():
            if m.__class__.__name__ in self.known_modules:
                v = updates[m]
                m.weight.grad.data.copy_(v[0])
                if v[1] is not None:
                    m.bias.grad.data.copy_(v[1])

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p.copy_(buf)

                p.data.add_(-group['lr'], d_p)


    def memory_usage(self):

        def sizeof_tensor(tensor):
            if isinstance(tensor, list) or isinstance(tensor, tuple):
                return sum([sizeof_tensor(t) for t in tensor])
            return (tensor.nelement() * tensor.element_size() 
                    if tensor is not None else 0)
        mem = 0
        for m in self.modules:
            factors = []
            if m in self.m_I:
                factors.append(self.m_I[m])
            if m in self.m_G:
                factors.append(self.m_G[m])
            if m in self.m_II:
                factors.append(self.m_II[m])
            if m in self.m_GG:
                factors.append(self.m_GG[m])
            if m in self.m_K:
                factors.append(self.m_K[m])
            mem += sizeof_tensor(factors)
        return mem

    @torch.no_grad()
    def reset_timing(self):
        self.timing['allgather_factors'] = 0
        self.timing['compute_factors'] = 0
        self.timing['compute_kernel'] = 0
        self.timing['update_inv'] = 0
        self.timing['broadcast_inv'] = 0
        self.timing['precondition'] = 0
        self.timing['memory_usage'] = 0
        self.timing['compute_projection'] = 0
        self.timing['id_time'] = 0

    @torch.no_grad()
    def step(self, closure=None):        

        if self.steps % self.freq == 0:
            self.allgather_factors()
            self.compute_factors()
            self.update_inv()
            self.broadcast_inv()

            if self.backend.size() > 1:
                self.timing['allgather_factors'] = self.timing['allgather_factors_comm'] + self.timing['allgather_factors_cat']
            
            if not self.randk:
                self.compute_projection()

            if self.steps >= self.damping_decay:
                alpha = 2 * torch.log10(torch.Tensor([self.init_damping / self.target_damping]).to(self.m_I[self.modules[0]].device)) / 50
                self.damping = (1 - alpha) * self.damping + alpha * self.target_damping
                self.damping = self.damping.item()
            mem = self.memory_usage()
            self.timing['memory_usage'] += mem
            
            if self.profiling:
                line = ''
                for k in self.timing:
                    line += str(self.timing[k])
                    line += ','
                line = line[:-1] + '\n'
                f = open(self.fname, 'a')
                f.write(line)
                f.close()

                self.reset_timing()

        if self.grad_norm and self.backend.rank() == 0:
            fname = 'grad-norm.csv'
            line = ''
            for m in self.modules:
                weight_norm = torch.norm(m.weight.grad.data)
                line += str(weight_norm.item())
                line += ', '
            line = line[:-1] + '\n'
            f = open(fname, 'a')
            f.write(line)
            f.close()

        if self.rank_analysis and self.backend.rank() == 0:
            X = []
            for k in self.ranks:
                X.append(k)

            fname = self.prefix + 'rank-' + str(self.backend.rank()) + '.csv'
            f = open(fname, 'a')
            for i in X:
                f.write(str(i)+',')
            f.write('\n')
            f.close()

        updates = self.precondition()
        self.update_grad_and_step(updates)

        self.steps += 1

        if not (self.sngd or self.enable_id or self.enable_is) and self.adaptive is not None:
            self.randk = not ((self.steps < self.iters_per_epoch * self.adaptive[0]) 
                or (self.steps > self.iters_per_epoch * self.adaptive[1] and self.steps <= self.iters_per_epoch * self.adaptive[2]))

