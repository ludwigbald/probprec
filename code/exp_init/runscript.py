"""Example run script for tuning using SORunner."""

import torch.optim as optim
from torch import nn
import deepobs.pytorch as pyt
from deepobs.pytorch.datasets.cifar10 import cifar10
from deepobs.pytorch.testproblems.testproblems_utils import tfconv2d
from deepobs.pytorch.testproblems.testproblems_utils import tfmaxpool2d
from deepobs.pytorch.testproblems.testproblems_utils import flatten


from sorunner import SORunner
from probprec import Preconditioner
import numpy
import math


class PreconditionedSGD(Preconditioner):
    """docstring for PreconditionedSGD"""
    def __init__(self, *args, **kwargs):
        super(PreconditionedSGD, self).__init__(*args, optim_class = optim.SGD, **kwargs)

class PreconditionedSGD_noinit(Preconditioner):
    """docstring for PreconditionedSGD"""
    def __init__(self, *args, **kwargs):
        super(PreconditionedSGD_noinit, self).__init__(*args, optim_class = optim.SGD, **kwargs)



# monkey patch: switch net for Filips version
class net_with_init(nn.Module):

    def __init__(self, num_classes=10):
        super(net_with_init, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5,padding=0),#, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Conv2d(64, 96, kernel_size=3,padding=0),#, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * 3 * 3)
        x = self.classifier(x)
        return x

def setup_with_init(self):
    self.data = cifar10(self._batch_size)
    self.loss_function = nn.CrossEntropyLoss
    self.net = net_with_init()
    self.net.to(self._device)
    self.regularization_groups = self.get_regularization_groups()

# monkey patch: switch net for Filips version
class net_no_init(nn.Module):

    def __init__(self, num_classes=10):
        super(net_no_init, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5,padding=0),#, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Conv2d(64, 96, kernel_size=3,padding=0),#, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        # DON'T init the layers
        # for module in self.modules():
        #     if isinstance(module, nn.Conv2d):
        #         nn.init.constant_(module.bias, 0.0)
        #         nn.init.xavier_normal_(module.weight)
        #
        #     if isinstance(module, nn.Linear):
        #         nn.init.constant_(module.bias, 0.0)
        #         nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * 3 * 3)
        x = self.classifier(x)
        return x

def setup_no_init(self):
    self.data = cifar10(self._batch_size)
    self.loss_function = nn.CrossEntropyLoss
    self.net = net_no_init()
    self.net.to(self._device)
    self.regularization_groups = self.get_regularization_groups()
    self.weight_decay = 0



# specify the Preconditioned Optimizer class
poptimizer_class = PreconditionedSGD

# and its hyperparameters
phyperparams = {'lr': {"type": float, 'default': None}}

# create the runner instance
prunner = SORunner(poptimizer_class, phyperparams)
pyt.testproblems.cifar10_3c3d.set_up = setup_with_init
prunner.run(testproblem='cifar10_3c3d', num_epochs = 5, batch_size = 32)


prunner = SORunner(PreconditionedSGD_noinit, phyperparams)
pyt.testproblems.cifar10_3c3d.set_up = setup_no_init
prunner.run(testproblem='cifar10_3c3d', num_epochs = 5, batch_size = 32)
