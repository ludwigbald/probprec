"""Example run script for tuning using SORunner."""

import torch.optim as optim
import deepobs.pytorch as pyt
from sorunner import SORunner
from probprec import Preconditioner
import numpy
import math

# DeepOBS setup


class PreconditionedSGDlog(Preconditioner):
    """docstring for PreconditionedSGD"""
    def __init__(self, *args, **kwargs):
        super(PreconditionedSGDlog, self).__init__(*args, optim_class = optim.SGD, **kwargs)

class PreconditionedAdam(Preconditioner):
    """docstring for PreconditionedAdam"""
    def __init__(self, *args, **kwargs):
        super(PreconditionedAdam, self).__init__(*args, optim_class = optim.Adam, **kwargs)




# specify the optimizer class
optimizer_class = optim.Adam

# and its hyperparameters
hyperparams = {'lr': {'type': float, 'default': 0.01}}


# specify the Preconditioned Optimizer class
poptimizer_class = PreconditionedSGDlog

# and its hyperparameters
phyperparams = {'lr': {"type": float, 'default': None}}

# create the runner instance
prunner = SORunner(poptimizer_class, phyperparams)

# prunner.run(testproblem='fmnist_2c2d', num_epochs = 1)

prunner.run(testproblem='quadratic_deep', batch_size = 32, num_epochs = 20)

# prunner.run(testproblem='mnist_vae', num_epochs=20, random_seed = 45)
#
# for lr in numpy.arange(-210.0, -320.0, -10.0):
#     for i in range(43, 44):
#         runner.run(testproblem='quadratic_deep', random_seed = i, batch_size = 32, hyperparams= {'lr': 10**lr})
#         prunner.run(testproblem='quadratic_deep', random_seed = i, batch_size = 32, hyperparams= {'lr': 10**lr})

# run the optimizer on a testproblem
#runner.run()
