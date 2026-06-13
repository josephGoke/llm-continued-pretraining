import torch
from torch.optim import AdamW



# ============================================================================
#  CHILD TUNING OPTIMIZER IMPLEMENTATION (TASK-FREE MODE)
# ============================================================================
class ChildTuningAdamW(AdamW):
    """
    AdamW optimizer modified for Task-Free Child Tuning.
    Randomly masks gradients and amplifies the active ones to emulate 
    fine-grained layer regularization without breaking FSDP sharding.
    """
    def __init__(self, params, lr=1e-3, reserve_p=0.30, mode="TaskFree", **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.reserve_p = reserve_p
        self.mode = mode

    def step(self, closure=None):
        if self.mode == "TaskFree":
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # 1. Create Bernoulli mask (prob = reserve_p)
                    prob = torch.full_like(p.grad, self.reserve_p)
                    mask = torch.bernoulli(prob)
                    
                    # 2. Apply mask and amplify gradients mathematically
                    p.grad.data.mul_(mask).div_(self.reserve_p)
        
        # 3. Proceed with standard AdamW optimization step
        return super().step(closure)