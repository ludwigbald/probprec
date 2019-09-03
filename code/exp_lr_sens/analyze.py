from deepobs import analyzer

# get the overall best performance of the MomentumOptimizer on the quadratic_deep testproblem
performance_dic = analyzer.get_performance_dictionary('./results/fmnist_2c2d/Preconditioner')
print(performance_dic)
performance_dic = analyzer.get_performance_dictionary('./results/fmnist_2c2d/PreconditionedSGD')
print(performance_dic)
# plot the training curve for the best performance
#analyzer.plot_optimizer_performance('./results/fmnist_2c2d/Preconditioner')

# lots all optimizer performances for all testproblems.
#analyzer.plot_testset_performances('./results/')

analyzer.plot_hyperparameter_sensitivity("./results/cifar10_3c3d/Preconditioner", hyperparam = 'lr')
analyzer.plot_hyperparameter_sensitivity("./results/fmnist_2c2d/Preconditioner", hyperparam = 'lr')
# analyzer.plot_hyperparameter_sensitivity("./results/mnist_vae/Preconditioner", hyperparam = 'lr')

# plot again, but this time compare to the Adam baseline
# analyzer.plot_optimizer_performance('./results/mnist_2c2d/Preconditioner2',
#                                     reference_path='./results/mnist_2c2d/SGD2')
