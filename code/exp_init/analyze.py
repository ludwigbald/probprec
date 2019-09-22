from deepobs import analyzer
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib
import codecs




# get the plot
fig, axess = analyzer.plot_testset_performances('./results/', mode = 'final')
axess[0][0].set_title("DeepOBS init")
axess[0][1].set_title("PyTorch default init")
axess[0][0].set_ylabel("test loss")
axess[1][0].set_ylabel("train loss")
axess[2][0].set_ylabel("test acc")
axess[3][0].set_ylabel("train acc")

axess[0][0].get_legend().remove()
axess[3][1].legend(["Batch Size = 32",
                 "Batch Size = 64",
                 "Batch Size = 128"])

#Change line styles
for axes in axess:
    lines = axes[0].get_lines()
    for line in lines[1:]:
        line.set_linewidth(3)
        line.set_linestyle("--")
        line.set_alpha(0.8)
    lines[0].set_linewidth(4)

    lines = axes[1].get_lines()
    for line in lines[1:]:
        line.set_linewidth(3)
        line.set_linestyle("--")
        line.set_alpha(0.8)
    lines[0].set_linewidth(4)


#Change plot y scales
axess[0][0].set_ylim(1, 4)
axess[1][0].set_ylim(1, 4)
axess[0][1].set_ylim(1, 4)
axess[1][1].set_ylim(1, 4)
axess[2][0].set_ylim(0.1, 0.75)
axess[3][0].set_ylim(0.1, 0.75)
axess[2][1].set_ylim(0.1, 0.75)
axess[3][1].set_ylim(0.1, 0.75)



# modify the plot


fig.canvas.draw()

# General settings
code = tikzplotlib.get_tikz_code(figure = fig,
                                 figurewidth = "\\figurewidth",
                                 figureheight = "5cm",
                                 extra_axis_parameters = ["tick pos=left",
             "legend style={font=\\footnotesize, at={(0 ,0)},xshift = -0.4cm, yshift=-1.5cm,anchor=north,nodes=right}",],
                                 extra_tikzpicture_parameters = ["every axis plot post./append style={line width = 1pt}"],
                                 )#strict = True)

#catch missed underscores & save
code = code.replace("\_", "_").replace("_", "\_")
file = codecs.open("../../thesis/images/exp_init.pgf", "w", 'utf-8')
file.write(code)
file.close()
