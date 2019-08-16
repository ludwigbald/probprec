from deepobs import analyzer
import json
import matplotlib.pyplot as plt

# get the overall best performance of the MomentumOptimizer on the quadratic_deep testproblem
performance_dic = analyzer.get_performance_dictionary('./results/quadratic_deep/PreconditionedSGDlog')
print(performance_dic)

# plot the training curve for the best performance
analyzer.plot_optimizer_performance('/home/ludwig/Desktop/probprec/code/results/quadratic_deep/PreconditionedSGDlog')

with open('/home/ludwig/Desktop/probprec/code/results/quadratic_deep/PreconditionedSGDlog/num_epochs__20__batch_size__32__lr__None/random_seed__42__2019-08-16-18-10-13.json', 'r') as f:
    results_dict = json.load(f)
plt.plot(results_dict['grad_norms'])
plt.show()


# plot again, but this time compare to the Adam baseline
# analyzer.plot_optimizer_performance('./results/quadratic_deep/PreconditionedSGD',
#                                     reference_path='./results/quadratic_deep/PreconditionedAdam')
