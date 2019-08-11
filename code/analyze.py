from deepobs import analyzer

# get the overall best performance of the MomentumOptimizer on the quadratic_deep testproblem
performance_dic = analyzer.get_performance_dictionary('./results/quadratic_deep/Preconditioner')
print(performance_dic)

# plot the training curve for the best performance
analyzer.plot_optimizer_performance('./results/quadratic_deep/Preconditioner')

# plot again, but this time compare to the Adam baseline
analyzer.plot_optimizer_performance('./results/quadratic_deep/Preconditioner',
                                    reference_path='./results/quadratic_deep/SGD')
analyzer.plot_hyperparameter_sensitivity('./results/quadratic_deep/', hyperparam = 'lr')
