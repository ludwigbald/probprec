from deepobs import analyzer
import json
import matplotlib.pyplot as plt



# plot the training curve for the best performance
analyzer.plot_optimizer_performance('./results64/cifar10_3c3d', reference_path = './results/cifar10_3c3d')
# plot the training curve for the best performance
analyzer.plot_optimizer_performance('./results/cifar10_3c3d')
