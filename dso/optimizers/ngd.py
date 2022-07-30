import torch

from math import ceil, floor
from torch import einsum, eye, inverse
from torch.optim import Optimizer
from ..utils.factors import ComputeI, ComputeG


class NGDOptimizer(Optimizer):
    def __init__(self, model, lr, init_damping, target_damping, damping_decay_steps, weight_decay, freq, momentum,
                 batch_size, iters, warmup_epochs, backend, compression_ratio=0.1, adaptive=None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=init_damping, weight_decay=weight_decay)

        super(NGDOptimizer, self).__init__(model.parameters(), defaults)

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
        self.damping_decay_steps = damping_decay_steps
        self.weight_decay = weight_decay
        self.freq = freq
        self.momentum = momentum

        self.global_bs = batch_size * backend.size()
        self.iters_per_epoch = iters
        self.damping_decay = self.iters_per_epoch * warmup_epochs

        self.backend = backend

        self.compression_ratio = compression_ratio
        self.adaptive = adaptive

        self.acc_stats = False
        self.steps = 0
        self.randk = not ((self.steps < self.iters_per_epoch * self.adaptive[0]) 
                or (self.steps > self.iters_per_epoch * self.adaptive[1] and self.steps <= self.iters_per_epoch * self.adaptive[2]))
        self._prepare_model()


    def _save_input(self, m, input):
        if torch.is_grad_enabled() and self.steps % self.freq == 0:
            self.m_I[m] = self.IHandler(input[0].data, m)


    def _save_grad_output(self, m, grad_input, grad_output):
        if self.acc_stats and self.steps % self.freq == 0:
            G, _ = self.GHandler(grad_output[0].data, m)
            self.m_G[m] = G
            if not self.randk:
                self.interpolative_decomposition(m)
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
            print('NGD keeps the following modules:')
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

        handles = []
        handles.extend(self._allgather_factors(X))
        self.backend.sync(handles)

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

    @torch.no_grad()
    def _compute_II(self, m):
        I = self.m_I[m]
        II = einsum('nk,qk->nq', (I, I))

        return II


    @torch.no_grad()
    def _compute_GG(self, m):
        G = self.m_G[m]
        GG = einsum('nk,qk->nq', (G, G))

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
        self.m_RDR[m] = RT @ Dinv @ RT.T
        
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

            K = (II * GG) / n
            if self.randk:
                self.m_K[m] = torch.inverse(K + self.damping * eye(n).to(II.device))
            else:
                self.m_K[m] = torch.inverse(K)


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

        handles = []
        for m in self.modules:
            handles.extend(self._broadcast_inv(m))
        self.backend.sync(handles)


    def compute_projection(self):
        for m in self.modules:
            if isinstance(self.m_RDR[m], list):
                RDR = torch.block_diag(*self.m_RDR[m])
            else:
                RDR = self.m_RDR[m]

            K = self.m_K[m]
            self.m_K[m] = (RDR - RDR @ K @ torch.inverse(torch.eye(K.shape[0]).to(K.device) + RDR @ K) @ RDR)

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
        updates = {}
        for m in self.modules:
            updates[m] = self._precondition(m)
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


    @torch.no_grad()
    def step(self, closure=None):
        if self.steps % self.freq == 0:
            self.allgather_factors()
            self.compute_factors()
            self.update_inv()
            self.broadcast_inv()
            if not self.randk:
                self.compute_projection()

            if self.steps >= self.damping_decay:
                alpha = 2 * torch.log10(torch.Tensor([self.init_damping / self.target_damping]).to(self.m_I[self.modules[0]].device)) / self.damping_decay_steps
                self.damping = (1 - alpha) * self.damping + alpha * self.target_damping
                self.damping = self.damping.item()

        updates = self.precondition()
        self.update_grad_and_step(updates)

        self.steps += 1

        if self.adaptive is not None:
            self.randk = not ((self.steps < self.iters_per_epoch * self.adaptive[0]) 
                    or (self.steps > self.iters_per_epoch * self.adaptive[1] and self.steps <= self.iters_per_epoch * self.adaptive[2]))

