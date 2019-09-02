from sorunner import SORunner
from probprec import Preconditioner
import torch.optim as optim
import deepobs.pytorch as pt
import numpy as np
import time

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

def estimate_runtime2(framework,
                     runner_cls,
                     optimizer_cls,
                     optimizer2_cls,
                     optimizer_hp,
                     n_runs = 5,
                     sgd_lr=0.01,
                     testproblem='mnist_mlp',
                     num_epochs = 5,
                     batch_size = 128,
                     **kwargs):

    # get the standard runner with SGD
    if framework == 'pytorch':
        from deepobs import pytorch as ptobs
        from torch.optim import SGD
        sgd_class = SGD
        hp = {'lr': {'type': float}}
        hyperparams = {"lr": sgd_lr}
        sgd_runner = pt.runners.StandardRunner(sgd_class, hp)

    else:
        raise RuntimeError('Framework must be pytorch or tensorflow')
    print("Correct version")
    sgd_times = []
    new_opt_times = []
    new_opt2_times = []

    for i in range(n_runs):
        print("** Start Run: ", i + 1, "of", n_runs)

        # SGD
        print("Running SGD")
        start_sgd = time.time()
        sgd_runner.run(
            testproblem=testproblem,
            hyperparams=hyperparams,
            batch_size=batch_size,
            num_epochs=num_epochs,
            no_logs=True,
            **kwargs)
        end_sgd = time.time()

        sgd_times.append(end_sgd - start_sgd)
        print("Time for SGD run ", i + 1, ": ", sgd_times[-1])

        # New Optimizer
        runner = runner_cls(optimizer_cls, optimizer_hp)
        print("Running...", optimizer_cls.__name__)
        start_script = time.time()
        runner.run(
            testproblem=testproblem,
            hyperparams=hyperparams,
            batch_size=batch_size,
            num_epochs=num_epochs,
            no_logs=True,
            **kwargs)
        end_script = time.time()

        new_opt_times.append(end_script - start_script)
        print("Time for new optimizer run ", i + 1, ": ", new_opt_times[-1])

        # New Optimizer 2
        runner2 = runner_cls(optimizer2_cls, optimizer_hp)
        print("Running...", optimizer2_cls.__name__)
        start_script = time.time()
        runner2.run(
            testproblem=testproblem,
            hyperparams=hyperparams,
            batch_size=batch_size,
            num_epochs=num_epochs,
            no_logs=True,
            **kwargs)
        end_script = time.time()

        new_opt2_times.append(end_script - start_script)
        print("Time for new optimizer 2 run ", i + 1, ": ", new_opt_times[-1])

    overhead = np.divide(new_opt_times, sgd_times)
    overhead2 = np.divide(new_opt2_times, sgd_times)

    output = "** Mean run time SGD: " + str(
        np.mean(sgd_times)) + "\n" + "** Mean run time new optimizer: " + str(
        np.mean(new_opt_times)) + "\n" + "** Overhead per run: " + str(
        overhead) + "\n" + "** Mean overhead: " + str(
        np.mean(overhead)) + " Standard deviation: " + str(
        np.std(overhead)) + "** Mean run time new optimizer 2: " + str(
        np.mean(new_opt2_times)) + "\n" + "** Overhead 2 per run: " + str(
        overhead2) + "\n" + "** Mean overhead 2: " + str(
        np.mean(overhead2)) + " Standard deviation 2: " + str(
        np.std(overhead2))


    print(output)
    return output


file = open("results.txt", "w+")

e = estimate_runtime2(framework = 'pytorch', runner_cls = SORunner,
                      optimizer_cls = AdaptiveSGD,
                      optimizer2_cls = PreconditionedSGD,
                      optimizer_hp = {},
                      testproblem = 'fmnist_2c2d')

file.write(e)
file.close()
