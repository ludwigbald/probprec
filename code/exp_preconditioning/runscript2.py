"""Simple run script using SORunner."""

import torch.optim as optim
import deepobs.pytorch as pyt
from sorunner import SORunner
from probprec import Preconditioner
import numpy
import math

# DeepOBS setup

class PreconditionedSGD(Preconditioner):
    """docstring for PreconditionedSGD"""
    def __init__(self, *args, **kwargs):
        super(PreconditionedSGD, self).__init__(*args, optim_class = optim.SGD, **kwargs)

class AdaptiveSGD(Preconditioner):
    """docstring for PreconditionedSGD"""
    def __init__(self, *args, **kwargs):
        super(AdaptiveSGD, self).__init__(*args, optim_class = optim.SGD, **kwargs)

    def _apply_preconditioner(self):
        return;


# specify the Preconditioned Optimizer class
poptimizer_class = PreconditionedSGD

# and its hyperparameters
phyperparams = {'lr': {"type": float, 'default': None}}

# create the runner instances
prunner = SORunner(PreconditionedSGD, phyperparams)

prunner.run(testproblem='fmnist_2c2d')
prunner.run(testproblem='cifar10_3c3d')
