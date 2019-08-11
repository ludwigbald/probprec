"""Example run script for tuning using SORunner."""

import torch.optim as optim
import deepobs.pytorch as pyt
from sorunner import SORunner
from probprec import Preconditioner
import numpy
import math

# DeepOBS setup


# specify the optimizer class
optimizer_class = optim.SGD
# and its hyperparameters
hyperparams = {'lr': {'type': float, 'default': 0.001}}
# create the runner instance
runner = pyt.runners.StandardRunner(optimizer_class, hyperparams)


# specify the Preconditioned Optimizer class
poptimizer_class = Preconditioner #Preconditioner
# and its hyperparameters
phyperparams = {'lr': {"type": float, 'default': 0.0774263682681127}}
# create the runner instance
prunner = SORunner(poptimizer_class, phyperparams)

prunner.run(testproblem='mnist_vae', num_epochs = 1)

#
# for lr in numpy.arange(-210.0, -320.0, -10.0):
#     for i in range(43, 44):
#         runner.run(testproblem='quadratic_deep', random_seed = i, batch_size = 32, hyperparams= {'lr': 10**lr})
#         prunner.run(testproblem='quadratic_deep', random_seed = i, batch_size = 32, hyperparams= {'lr': 10**lr})

# run the optimizer on a testproblem
#runner.run()
