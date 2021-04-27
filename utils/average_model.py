import torch
import torch.nn as nn

import copy
import warnings


class running_ensemble(nn.Module):
    def __init__(self, model):
        super(running_ensemble, self).__init__()
        self.model = copy.deepcopy(model)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.register_buffer('num_models', torch.zeros(1))
        self.bn_updated = False
        return

    def update(self, model):
        alpha = 1 / (self.num_models + 1)
        for p1, p2 in zip(self.model.parameters(), model.parameters()):
            p1.data *= (1 - alpha)
            p1.data += p2.data * alpha

        self.num_models += 1
        self.bn_update = False
    
    @staticmethod
    def _reset_bn(module):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)

    @staticmethod
    def _get_momenta(module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum


    @staticmethod
    def _set_momenta(module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]

    def update_bn(self, loader):
        self.model.train()
        self.model.apply(running_ensemble._reset_bn)
        is_cuda = next(self.model.parameters()).is_cuda

        momenta = {}
        self.model.apply(lambda module: running_ensemble._get_momenta(module, momenta))
        n = 0
        for X, _ in loader:
            if is_cuda:
                X = X.cuda()

            b = len(X)
            momentum = b / (n + b)

            for module in momenta.keys():
                module.momentum = momentum

            self.model(X)

            n += b

        self.model.apply(lambda module: running_ensemble._set_momenta(module, momenta))
        self.model.eval()
        self.bn_updated = True
        return 

    def forward(self, x):
        if not self.bn_updated:
            warnings.warn('Running Mean and Variance of BatchNorm is not Updated!. Use with Care!')
        return self.model(x)
