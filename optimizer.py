from typing import Callable, Iterable, Tuple

import torch
import math
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # https://stackoverflow.com/questions/51387194/implementing-adam-in-pytorch
                # State should be stored in this dictionary
                state = self.state[p]

                # Initialize the state
                if len(state) == 0:
                    state['step'] = 0
                    
                    # Initialize momentum vectors
                    state['m_t'] = torch.zeros_like(p.data)
                    state['v_t'] = torch.zeros_like(p.data)
                    state['alpha'] = group['lr']

                m_t, v_t = state['m_t'], state['v_t']

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta_1, beta_2 = group["betas"]
                epsilon = group["eps"]
                weight_decay = group['weight_decay']

                # Increase the step in the state
                state['step'] += 1

                # Update first and second moments of the gradients
                m_t = beta_1 * m_t + (1 - beta_1) * grad
                v_t = beta_2 * v_t + (1 - beta_2) * (grad * grad)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                state['alpha'] = (alpha * math.sqrt(1 - beta_2**state['step'])) / (1 - beta_1**state['step'])

                # Update parameters
                p.data = p.data - (((state['alpha'] * m_t) / (torch.sqrt(v_t) + epsilon)))

                # Add weight decay after the main gradient-based updates.
                p.data = p.data - weight_decay * alpha * p.data
                # Please note that the learning rate should be incorporated into this update.
                
                state['m_t'] = m_t
                state['v_t'] = v_t

        return loss