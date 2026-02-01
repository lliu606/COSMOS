import math
import torch
from torch import Tensor
from typing import List
from torch.optim import Optimizer

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


def cosmos(params: List[Tensor],
            grads: List[Tensor],
            exp_avgs: List[Tensor],
            exp_avg_sqs: List[Tensor],
            exp_avgs_GG: List[Tensor],
            exp_avgs_P: List[Tensor],
            max_exp_avg_sqs: List[Tensor],
            state_steps: List[int],
            *,
            amsgrad: bool,
            beta1: float,
            beta2: float,
            lr: float,
            weight_decay: float,
            eps: float,
            maximize: bool,
            ratio:float,
            ):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_gg = exp_avgs_GG[i]
        exp_avg_p = exp_avgs_P[i]

        step = state_steps[i]

        bias_correction1 = (1 - beta1 ** step) / (1 - beta1)
        bias_correction2 = 1 - beta2 ** step


        if len(param.size()) == 2 and param.size(0) <= 10000:
            exp_avg.mul_(beta1).add_(grad)

            if step == 1:
                W = torch.matmul(grad.T, grad)
                U, _, _ = torch.linalg.svd(W, full_matrices=False)
                exp_avg_p.data = U[:, :exp_avg_gg.size(0)]
                exp_avg_gg = torch.matmul(torch.matmul(exp_avg_p.T, grad.T), torch.matmul(grad, exp_avg_p)) * (1 - beta2)

            else:
                t = exp_avg_p.detach().clone().T
                exp_avg_p = beta2 * torch.matmul(exp_avg_p, exp_avg_gg) + (1-beta2)*torch.matmul(grad.T, torch.matmul(grad, exp_avg_p))
                exp_avg_p, _ = torch.linalg.qr(exp_avg_p, mode='reduced')
                t = torch.matmul(t, exp_avg_p)
                exp_avg_gg = beta2 * torch.matmul(t.T, torch.matmul(exp_avg_gg, t)) + (1-beta2)*torch.matmul(torch.matmul(grad, exp_avg_p).T, torch.matmul(grad, exp_avg_p))

            scale = (grad.size(0) * grad.size(1))**0.5 
            low_rank_grad = torch.matmul(grad, exp_avg_p)
            exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad.conj(), value=1 - beta2)

            grad.add_(exp_avg, alpha=beta1)
            grad.mul_(1 / (1 + beta1 * bias_correction1))

            t = torch.matmul(grad, exp_avg_p)
            step_size = lr / bias_correction1
            t1 = t/((exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)) # * step_size
            t1 = torch.matmul(t1, exp_avg_p.T)

        
            t = grad - torch.matmul(t, exp_avg_p.T)
            t = zeropower_via_newtonschulz5(t, steps=5)
            t = t/(t.norm() + eps)

            t1.add_(t, alpha = scale * 0.2)

            if weight_decay > 0:
                param.data.mul_(1-lr*weight_decay)

            param.add_(t1/(t1.norm() + eps), alpha=-scale * ratio * lr)

            exp_avgs_P[i].copy_(exp_avg_p)
            exp_avgs_GG[i].copy_(exp_avg_gg)

        else:
            bias_correction1 = 1 - 0.9 ** step
            exp_avg.mul_(beta1).add_(grad, alpha=0.1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            if weight_decay > 0:
                param.data.mul_(1-lr*weight_decay)
                
            param.addcdiv_(exp_avg, denom, value=-step_size)


class COSMOS(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, lr_ratio=0.1, rank=64,
                 weight_decay=0, amsgrad=False, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super(COSMOS, self).__init__(params, defaults)
        self.lr_ratio = lr_ratio
        self.rank = rank

    def __setstate__(self, state):
        super(COSMOS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avgs_GG = []
            exp_avgs_P = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values

                        if len(p.size()) == 2 and p.size(0) <= 10000:
                            state['exp_avg_GG'] = torch.zeros(self.rank, self.rank, dtype=p.data.dtype, device=p.data.device)
                            state['exp_avg_P'] = torch.zeros(p.size(1), self.rank, dtype=p.data.dtype, device=p.data.device)
                            state['exp_avg_sq'] = torch.zeros(p.size(0), self.rank, dtype=p.data.dtype, device=p.data.device)

                        else:
                            state['exp_avg_GG'] = torch.zeros(0)
                            state['exp_avg_P'] = torch.zeros(0)
                            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    exp_avgs_GG.append(state['exp_avg_GG'])
                    exp_avgs_P.append(state['exp_avg_P'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            cosmos(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avgs_GG,
                exp_avgs_P,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                ratio=self.lr_ratio,
                )

        return loss