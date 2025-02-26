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


def soap_with_muon(params: List[Tensor],
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
            beta3: float,
            lr: float,
            weight_decay: float,
            eps: float,
            maximize: bool,
            ratio:float,
            ):

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_gg = exp_avgs_GG[i]
        exp_avg_p = exp_avgs_P[i]

        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        bias_correction3 = 1 - beta3 ** step

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if len(param.size()) == 2 and param.size(0) <= 10000:
            if step == 1:
                W = torch.matmul(grad, grad.T)
                U, _, _ = torch.linalg.svd(W, full_matrices=False)
                exp_avg_p.data = U[:, :exp_avg_gg.size(0)]
                exp_avg_gg = torch.matmul(torch.matmul(exp_avg_p.T, W), exp_avg_p) * (1 - beta3)
            else:
                t = exp_avg_p.detach().clone().T
                exp_avg_p = beta3 * torch.matmul(exp_avg_p, exp_avg_gg) + (1-beta3)*torch.matmul(grad, torch.matmul(grad.T, exp_avg_p))
                exp_avg_p, _ = torch.linalg.qr(exp_avg_p, mode='reduced')
                t = torch.matmul(t, exp_avg_p)
                exp_avg_gg = beta3 * torch.matmul(t.T, torch.matmul(exp_avg_gg, t)) + (1-beta3)*torch.matmul(torch.matmul(grad.T, exp_avg_p).T, torch.matmul(grad.T, exp_avg_p))

            scale = (grad.size(0) * grad.size(1))**0.5
            low_rank_grad = torch.matmul(exp_avg_p.T, grad)
            exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad.conj(), value=1 - beta2)

            t = torch.matmul(exp_avg_p.T, exp_avg)
            step_size = lr / bias_correction1
            t1 = step_size * torch.matmul(exp_avg_p, t/((exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)))
            t = exp_avg - torch.matmul(exp_avg_p, t)
            
            if t.size(1) == 3 * t.size(0):
                t = torch.cat([zeropower_via_newtonschulz5(g1, steps=5) for g1 in t.split(t.size(0), dim=1)], dim=1)
            else:
                t = zeropower_via_newtonschulz5(t, steps=5)

            t = t/(t.norm() + eps)
            t1.add_(t, alpha=scale * ratio * lr)
            param.add_(t1/(t1.norm() + eps), alpha=-scale * ratio * lr)

        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1
            param.addcdiv_(exp_avg, denom, value=-step_size)


class COSMOS_for_llama(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.96, 0.96), eps=1e-8, lr_ratio=0.1, rank=64,
                 weight_decay=0, amsgrad=False, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super(COSMOS_for_llama, self).__init__(params, defaults)
        self.lr_ratio = lr_ratio
        self.rank = rank

    def __setstate__(self, state):
        super(COSMOS_for_llama, self).__setstate__(state)
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
            beta1, beta2, beta3 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'COSMOS does not support sparse gradients.')
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
                            state['exp_avg_P'] = torch.zeros(p.size(0), self.rank, dtype=p.data.dtype, device=p.data.device)
                            state['exp_avg_sq'] = torch.zeros(self.rank, p.size(1), dtype=p.data.dtype, device=p.data.device)

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

            soap_with_muon(
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
                beta3=beta3,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                ratio=self.lr_ratio,
                )

        return loss
