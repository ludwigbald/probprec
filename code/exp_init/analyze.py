from deepobs import analyzer
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib
import codecs




# get the plot
fig, axess = analyzer.plot_optimizer_performance('./results64/cifar10_3c3d',
                                reference_path = './results/cifar10_3c3d')

axess[0].set_ylabel("test loss")
axess[1].set_ylabel("train loss")
axess[2].set_ylabel("test acc")
axess[3].set_ylabel("train acc")

fig.legend(labels = ["1", "2", "3", "4"])

# modify the plot
for axes in axess:
    break

axess[0].get_legend().set_labels(["1", "2", "3", "4"])

mpl.use("pgf")

mpl.rcParams["pgf.rcfonts"] = False
fig.set_figwidth(4)
fig.set_figheight(6)
fig.savefig("../../thesis/images/exp_init.pgf")


# # General settings
# code = tikzplotlib.get_tikz_code(figure = fig,
#                                  figurewidth = "\linewidth",
#                                  figureheight = "7cm",
#                                  extra_axis_parameters = ["legend cell align = right",
#                                                           "tick pos=left",
#                                                           "xlabel"
#                                                           ],
#                                  extra_tikzpicture_parameters = [],
#                                  strict = True)
#
# #catch missed underscores & save
# code = code.replace("\_", "_").replace("_", "\_")
# file = codecs.open("../../thesis/images/exp_init.tex", "w", 'utf-8')
# file.write(code)
# file.close()
