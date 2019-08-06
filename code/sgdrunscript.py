"""Example run script using StandardRunner."""

from torch.optim import SGD
from deepobs import pytorch as pt

optimizer_class = SGD
hyperparams = {"lr": {"type": float, "default": 0.01},
               "nesterov": {"type": bool, "default": False}}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
runner.run(testproblem='mnist_2c2d')
