"""Example run script using SORunner."""

import torch.optim as optim
import deepobs.pytorch as pyt
from sorunner import SORunner
from probprec import Preconditioner

# DeepOBS setup


# specify the optimizer class
optimizer_class = optim.SGD

# and its hyperparameters
hyperparams = {'lr': {'type': float, 'default': 0.01},
               'momentum': {'type': float, 'default': 0.99}}


# specify the Preconditioned Optimizer class
poptimizer_class = Preconditioner

# and its hyperparameters
phyperparams = {'lr': {"type": float, "default": 0.01}}
# 'optim_class': optimizer_class,
# 'optim_hyperparams': hyperparams}

# create the runner instance
runner = SORunner(poptimizer_class, phyperparams)

# runner.run(testproblem='quadratic_deep', batch_size = 1, num_epochs=10)

# run the optimizer on a testproblem
runner.run(testproblem='mnist_2c2d')
