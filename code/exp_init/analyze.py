from deepobs import analyzer
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib
import codecs




# get the plot
fig, axess = analyzer.plot_optimizer_performance('./results64/cifar10_3c3d',
                                reference_path = './results/cifar10_3c3d')
axess[0].get_legend().remove()
axess[3].legend(["pytorch init, bs = 64",
                 "DeepOBS init, bs = 64",
                 "pytorch init, bs = 128",
                 "DeepOBS init, bs = 128"])
axess[0].set_title("")

axess[0].set_ylabel("test loss")
axess[1].set_ylabel("train loss")
axess[2].set_ylabel("test acc")
axess[3].set_ylabel("train acc")

axess


# modify the plot
for axes in axess:
    lines = axes.get_lines()
    for line in lines:
        line.set_linewidth(3)
#
#
# mpl.use("pgf")
#
# mpl.rcParams["pgf.rcfonts"] = False
# fig.set_figwidth(4)
# fig.set_figheight(6)
# fig.savefig("../../thesis/images/exp_init.pgf")


# General settings
code = tikzplotlib.get_tikz_code(figure = fig,
                                 figurewidth = "\\figurewidth",
                                 figureheight = "4cm",
                                 extra_axis_parameters = ["tick pos=left",
             "legend style={font=\\footnotesize, at={(0.5 ,0)},yshift=-1.5cm,anchor=north,nodes=right}",],
                                 extra_tikzpicture_parameters = ["every axis plot post./append style={line width = 1pt}"],
                                 )#strict = True)

#catch missed underscores & save
code = code.replace("\_", "_").replace("_", "\_")
file = codecs.open("../../thesis/images/exp_init.pgf", "w", 'utf-8')
file.write(code)
file.close()
