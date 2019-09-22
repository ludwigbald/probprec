"""Simple run script using SORunner."""

import torch.optim as optim
import deepobs.pytorch as pyt
from sorunner import SORunner
from probprec import Preconditioner
import numpy
import math


class PreconditionedSGD(Preconditioner):
    """docstring for PreconditionedSGD"""
    def __init__(self, *args, **kwargs):
        super(PreconditionedSGD, self).__init__(*args, optim_class = optim.SGD, **kwargs)


# Preconditioned SGD, but without
class TunedFmnistPreconditionedSGD(Preconditioner):
    def __init__(self, *args, **kwargs):
        super(TunedFmnistPreconditionedSGD, self).__init__(*args, optim_class = optim.SGD, **kwargs)

    def _init_the_optimizer(self):
        for group in self.param_groups:
            group.update(lr=0.11288378916846883)
            print("[_init_the_optimizer] Group Learning Rate:", group['lr'])
        self.optim_hyperparams.pop("lr", None)

        print("[_init_the_optimizer] Initializing ", self.optim_class.__name__, " with: ", self.optim_hyperparams)
        self.the_optimizer = self.optim_class(
            self.param_groups, **self.optim_hyperparams)


# Preconditioned SGD, but without
class TunedCifarPreconditionedSGD(Preconditioner):
    def __init__(self, *args, **kwargs):
        super(TunedCifarPreconditionedSGD, self).__init__(*args, optim_class = optim.SGD, **kwargs)

    def _init_the_optimizer(self):
        for group in self.param_groups:
            group.update(lr=0.04832930238571752)
            print("[_init_the_optimizer] Group Learning Rate:", group['lr'])
        self.optim_hyperparams.pop("lr", None)

        print("[_init_the_optimizer] Initializing ", self.optim_class.__name__, " with: ", self.optim_hyperparams)
        self.the_optimizer = self.optim_class(
            self.param_groups, **self.optim_hyperparams)




# and its hyperparameters for correct file naming, these are the optimal learning rates for SGD from the baselines
hyperparams_fmnist = {'lr': {"type": float, 'default':  0.11288378916846883}}

hyperparams_cifar = {'lr': {"type": float, 'default': 0.04832930238571752}}

# create the runner instances
frunner = SORunner(TunedFmnistPreconditionedSGD, hyperparams_fmnist)
# create the runner instances
crunner = SORunner(TunedCifarPreconditionedSGD, hyperparams_cifar)

frunner.run(testproblem='fmnist_2c2d')
crunner.run(testproblem='cifar10_3c3d')
