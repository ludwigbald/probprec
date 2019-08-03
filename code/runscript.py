"""Example run script using SORunner."""

import torch.optim as optim
import deepobs.pytorch as pyt
from sorunner import SORunner
from probprec import Preconditioner

# specify the optimizer class
optimizer_class = optim.SGD

# and its hyperparameters
hyperparams = {'lr': {'type': float, 'default': 0.01},
               'momentum': {'type': float, 'default': 0.99}}

# specify the Preconditioned Optimizer class
poptimizer_class = Preconditioner

# and its hyperparameters
phyperparams = {'est_rank': {"type": int, "default": 2},
                'num_observations': {"type": int, "default": 5},
                'prior_iterations': {"type": int, "default": 5},
                'lr': {"type": float, "default": 0.01}}
                # 'weight_decay' : {"type" : float, "default": 0.999}
               # }
                #'optim_class': optimizer_class,
                #'optim_hyperparams': hyperparams}

# create the runner instance
runner = SORunner(poptimizer_class, phyperparams)

# runner.run(testproblem='quadratic_deep', batch_size = 1, num_epochs=10)

# run the optimizer on a testproblem
runner.run(testproblem ='mnist_2c2d',
            batch_size=128,
            num_epochs=1)

# run the Poptimizer again with different settings (estimate the lr)
phyperparams = {'est_rank': {"type": int, "default": 2},
                'num_observations': {"type": int, "default": 5},
                'prior_iterations': {"type": int, "default": 5}}
runner = SORunner(poptimizer_class, phyperparams)

runner.run(testproblem ='mnist_2c2d',
            batch_size=128,
            num_epochs=1)
