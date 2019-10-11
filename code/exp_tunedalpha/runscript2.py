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
class TunedFmnistPreconditionedMomentum(Preconditioner):
    def __init__(self, *args, **kwargs):
        super(TunedFmnistPreconditionedMomentum, self).__init__(*args, optim_class = optim.SGD, **kwargs)

    def _init_the_optimizer(self):
        for group in self.param_groups:
            group.update(lr=0.02069138081114788) #TODO: Get learning rate
            print("[_init_the_optimizer] Group Learning Rate:", group['lr'])
        self.optim_hyperparams.pop("lr", None)

        print("[_init_the_optimizer] Initializing ", self.optim_class.__name__, " with: ", self.optim_hyperparams)
        self.the_optimizer = self.optim_class(
            self.param_groups, **self.optim_hyperparams)


# Preconditioned SGD, but without
class TunedCifarPreconditionedMomentum(Preconditioner):
    def __init__(self, *args, **kwargs):
        super(TunedCifarPreconditionedMomentum, self).__init__(*args, optim_class = optim.SGD, **kwargs)

    def _init_the_optimizer(self):
        for group in self.param_groups:
            group.update(lr=0.00379269019073225) #TODO: Get learning rate
            print("[_init_the_optimizer] Group Learning Rate:", group['lr'])
        self.optim_hyperparams.pop("lr", None)

        print("[_init_the_optimizer] Initializing ", self.optim_class.__name__, " with: ", self.optim_hyperparams)
        self.the_optimizer = self.optim_class(
            self.param_groups, **self.optim_hyperparams)




# and its hyperparameters for correct file naming, these are the optimal learning rates for SGD from the baselines
hyperparams_fmnist = {'lr': {"type": float, 'default':  0.02069138081114788},
			'momentum' : {"type": float, 'default': 0.9}}

hyperparams_cifar = {'lr': {"type": float, 'default': 0.00379269019073225},
			'momentum' : {"type": float, 'default': 0.9}}

# create the runner instances
frunner = SORunner(TunedFmnistPreconditionedMomentum, hyperparams_fmnist)
# create the runner instances
crunner = SORunner(TunedCifarPreconditionedMomentum, hyperparams_cifar)

frunner.run(testproblem='fmnist_2c2d')
crunner.run(testproblem='cifar10_3c3d')
