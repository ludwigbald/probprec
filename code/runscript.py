"""Example run script for tuning using SORunner."""

import torch.optim as optim
import deepobs.pytorch as pyt
from sorunner import SORunner
from probprec import Preconditioner

# DeepOBS setup


# specify the optimizer class
#optimizer_class = optim.SGD

# and its hyperparameters
#hyperparams = {'lr': {'type': float}}


# specify the Preconditioned Optimizer class
poptimizer_class = Preconditioner

# and its hyperparameters
phyperparams = {} #'lr': {"type": float, 'default': None}}

# create the runner instance
runner = SORunner(poptimizer_class, phyperparams)

runner.run(testproblem='mnist_2c2d', num_epochs=2)

# run the optimizer on a testproblem
#runner.run()
