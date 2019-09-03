import numpy as np
from deepobs.pytorch.runners import StandardRunner
from deepobs.tuner import GridSearch
from torch.optim import SGD
from probprec import Preconditioner
from sorunner import SORunner

optimizer_class = Preconditioner
hyperparams = {"lr": {"type": float},
            "est_rank": {"type": int}}

# The discrete values to construct a grid for.
grid = {'lr': np.logspace(-5, 2, 10),
        'est_rank': [2,3]}

# Make sure to set the amount of ressources to the grid size. For grid search, this is just a sanity check.
tuner = GridSearch(optimizer_class, hyperparams, grid,
                   runner=SORunner, ressources=20)

# Tune (i.e. evaluate every grid point) and rerun the best setting with 10 different seeds.
# tuner.tune('quadratic_deep', rerun_best_setting=True, num_epochs=2, output_dir='./grid_search')

# Optionally, generate commands for a parallelized execution
tuner.generate_commands_script('mnist_vae', run_script='/home/bald/pre_fmnist/runscript.py',
                               output_dir='./grid_search', generation_dir='./grid_search_commands')
