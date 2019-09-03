from deepobs import analyzer
import json
import matplotlib.pyplot as plt

class Overhead:
    def __init__(self, name, overheads, mean, std):
        self.name = name
        self.overheads = overheads
        self.mean = mean
        self.std = std

#Manually transcribing the results from the results file

sgd = Overhead(name = 'SGD',
               overheads = [1, 1, 1, 1, 1],
               mean = 1,
               std = 0)
precon_sgd = Overhead(name = 'PreconditionedSGD',
                      overheads = [1.87697424, 2.85373601, 2.73309405, 2.23167897, 2.24178113],
                      mean = 2.387452881992991,
                      std = 0.3585978215624523)
adapt_sgd = Overhead(name = 'AdaptiveSGD',
                      overheads = [1.49414586, 2.18251235, 2.12219107, 1.74136945, 1.80575808],
                      mean = 1.8691953628606874,
                      std = 0.2542440740178504)
adapt_only_sgd = Overhead(name = 'OnlyAdaptiveSGD',
                      overheads = [0.85414052, 1.04404432, 1.01982744, 1.03676231, 1.03591921],
                      mean = 0.9981387614088575,
                      std = 0.07243252473207047)
object_list = [sgd, adapt_only_sgd, adapt_sgd, precon_sgd]

plt.figure(figsize= (3,3))
plt.bar([o.name for o in object_list],
        [o.mean for o in object_list], alpha = 0.7,
        yerr = [o.std for o in object_list], capsize = 10)
plt.hlines(1, xmin = -1, xmax = 4, linestyles= 'dashed', alpha = 0.7)
plt.show()


# plot again, but this time compare to the Adam baseline
# analyzer.plot_optimizer_performance('./results/quadratic_deep/PreconditionedSGD',
#                                     reference_path='./results/quadratic_deep/PreconditionedAdam')
