from deepobs import analyzer
import codecs

import tikzplotlib
import matplotlib as mpl
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
precon_sgd = Overhead(name = 'Preconditioned SGD',
                      overheads = [1.87697424, 2.85373601, 2.73309405, 2.23167897, 2.24178113],
                      mean = 2.387452881992991,
                      std = 0.3585978215624523)
adapt_sgd = Overhead(name = 'Adaptive SGD',
                      overheads = [1.49414586, 2.18251235, 2.12219107, 1.74136945, 1.80575808],
                      mean = 1.8691953628606874,
                      std = 0.2542440740178504)
adapt_only_sgd = Overhead(name = 'Only Adaptive SGD',
                      overheads = [0.85414052, 1.04404432, 1.01982744, 1.03676231, 1.03591921],
                      mean = 0.9981387614088575,
                      std = 0.07243252473207047)
object_list = [sgd, adapt_only_sgd, adapt_sgd, precon_sgd]




fig = plt.figure(figsize= (3,3))
plt.bar([o.name for o in object_list],
        [o.mean for o in object_list], alpha = 0.7,
        yerr = [o.std for o in object_list], capsize = 7)
plt.hlines(1, xmin = -1, xmax = 4, linestyles= 'dashed', alpha = 0.7)
plt.xticks(rotation = '25', horizontalalignment = 'right', verticalalignment = 'top')
plt.show()


# mpl.use("pgf")
# mpl.rcParams["pgf.rcfonts"] = False
# plt.gcf().savefig("../../thesis/images/exp_perf_prec.pgf")


# General settings
code = tikzplotlib.get_tikz_code(figure = fig,
                                 extra_axis_parameters = ["tick pos=left",
            "xticklabel style = {anchor = north east}"],
                                 )#strict = True)
print(code)
#catch missed underscores & save
code = code.replace("\_", "_").replace("_", "\_")
file = codecs.open("../../thesis/images/exp_perf_prec.pgf", "w", 'utf-8')
file.write(code)
file.close()
