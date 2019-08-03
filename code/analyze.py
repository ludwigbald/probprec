from deepobs import analyzer

# get the overall best performance of the MomentumOptimizer on the quadratic_deep testproblem
performance_dic = analyzer.get_performance_dictionary('./results/mnist_2c2d/Preconditioner')
print(performance_dic)

# plot the training curve for the best performance
#analyzer.plot_optimizer_performance('./results/mnist_2c2d/Preconditioner')

# plot again, but this time compare to the Adam baseline
analyzer.plot_optimizer_performance('./results/mnist_2c2d/Preconditioner',
                                    reference_path='./results/mnist_2c2d/Preconditioner2')
