from deepobs import analyzer
import json
import matplotlib.pyplot as plt
import tikzplotlib
import codecs

fig, axess = analyzer.plot_testset_performances("./results", reference_path = "../baselines/", mode = 'final')

axess[0][0].get_legend().remove()
axess[3][1].legend()

axess[1][0].set_ylim(0.05, 0.25)
axess[2][0].set_ylim(0.85, 0.93)
axess[3][0].set_ylim(0.85, 1,0)

axess[0][0].set_ylabel("test loss")
axess[1][0].set_ylabel("train loss")
axess[2][0].set_ylabel("test acc")
axess[3][0].set_ylabel("train acc")



# modify the plot
for axes in axess:
    for ax in axes:
        lines = ax.get_lines()
        for line in lines:
            line.set_linewidth(3)

fig.canvas.draw()
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
                                 figureheight = "5cm",
                                 extra_axis_parameters = ["tick pos=left",
             "legend style={font=\\footnotesize, at={(0 ,0)},xshift=-0.4cm, yshift=-1.5cm,anchor=north,nodes=right}",],
                                 extra_tikzpicture_parameters = ["every axis plot post./append style={line width = 1pt}"],
                                 )#strict = True)

#catch missed underscores & save
code = code.replace("\_", "_").replace("_", "\_")
file = codecs.open("../../thesis/images/exp_preconditioning.pgf", "w", 'utf-8')
file.write(code)
file.close()

################### Presentation


fig.canvas.draw()

code = tikzplotlib.get_tikz_code(figure = fig,
                                 figurewidth = "\\figurewidth",
                                 figureheight = "4cm",
                                 extra_axis_parameters = ["tick pos=left",
             "legend style={font=\\footnotesize, at={(0 ,0)},xshift=-0.4cm, yshift=-1.5cm,anchor=north,nodes=right}",],
                                 extra_tikzpicture_parameters = ["every axis plot post./append style={line width = 1pt}"],
                                 )#strict = True)

#catch missed underscores & save
code = code.replace("\_", "_").replace("_", "\_")
file = codecs.open("../../presentation/images/exp_preconditioning.pgf", "w", 'utf-8')
file.write(code)
file.close()

################## Presentation part 2
# get rid of upper two figures

fig.delaxes(axess[0][0])
fig.delaxes(axess[0][1])
fig.delaxes(axess[1][0])
fig.delaxes(axess[1][1])
fig.canvas.draw()

code = tikzplotlib.get_tikz_code(figure = fig,
                                 figurewidth = "\\figurewidth",
                                 figureheight = "4cm",
                                 extra_axis_parameters = ["tick pos=left",
             "legend style={font=\\footnotesize, at={(0 ,0)},xshift=-0.4cm, yshift=-1.5cm,anchor=north,nodes=right}",],
                                 extra_tikzpicture_parameters = ["every axis plot post./append style={line width = 1pt}"],
                                 )#strict = True)

#catch missed underscores & save
code = code.replace("\_", "_").replace("_", "\_")
file = codecs.open("../../presentation/images/exp_preconditioning2.pgf", "w", 'utf-8')
file.write(code)
file.close()
