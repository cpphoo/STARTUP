import torch.nn as nn


class dataparallel_wrapper(nn.Module):
    def __init__(self, module):
        super(dataparallel_wrapper, self).__init__()
        self.module = module

    def forward(self, mode, *args, **kwargs):
        return getattr(self.module, mode)(*args, **kwargs)
