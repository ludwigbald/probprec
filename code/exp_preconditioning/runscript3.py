"""Simple run script using SORunner."""
import numpy as np
import torch.optim as optim
import deepobs.pytorch as pyt
from deepobs.tuner import GridSearch
from sorunner import SORunner
from probprec import Preconditioner
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

runner = SORunner(AdaptiveSGD, phyperparams)

for rs in range(42,52):
    prunner.run(testproblem='quadratic_deep', num_epochs=20, batch_size = 32, random_seed = rs)
    runner.run(testproblem='quadratic_deep', num_epochs=20, batch_size = 32, random_seed = rs)


## SGD on quadratic deep

optimizer_class = optim.SGD
hyperparams = {"lr": {"type": float}}

# The discrete values to construct a grid for.
grid = {'lr': np.logspace(-5, 2, 10)}

# Make sure to set the amount of ressources to the grid size. For grid search, this is just a sanity check.
tuner = GridSearch(optimizer_class, hyperparams, grid, runner=pyt.runners.StandardRunner, ressources=6*3*2)

# Tune (i.e. evaluate every grid point) and rerun the best setting with 10 different seeds.
tuner.tune('quadratic_deep', rerun_best_setting=True, num_epochs=20, batch_size = 32, output_dir='./results')
