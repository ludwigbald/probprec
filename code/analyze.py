from deepobs import analyzer

# get the overall best performance of the MomentumOptimizer on the quadratic_deep testproblem
performance_dic = analyzer.get_performance_dictionary('./results/quadratic_deep/PreconditionedSGD')
print(performance_dic)

# plot the training curve for the best performance
analyzer.plot_optimizer_performance('./results/quadratic_deep/PreconditionedSGD')

# plot again, but this time compare to the Adam baseline
analyzer.plot_optimizer_performance('./results/quadratic_deep/PreconditionedSGD',
                                    reference_path='./results/quadratic_deep/PreconditionedAdam')
