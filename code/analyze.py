from deepobs import analyzer

# get the overall best performance of the MomentumOptimizer on the quadratic_deep testproblem
performance_dic = analyzer.get_performance_dictionary('./results/mnist_2c2d/Preconditioner2')
print(performance_dic)

# plot the training curve for the best performance
analyzer.plot_optimizer_performance('./results/mnist_2c2d/Preconditioner2')

# plot again, but this time compare to the Adam baseline
analyzer.plot_optimizer_performance('./results/mnist_2c2d/Preconditioner2',
                                    reference_path='./results/mnist_2c2d/SGD2')
