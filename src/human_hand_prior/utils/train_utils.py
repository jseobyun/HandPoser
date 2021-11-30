import os
import numpy as np
import random
import torch
import torch.nn as nn

def make_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class geodesic_loss_R(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(geodesic_loss_R, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch)))
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))

        else:
            return theta