from deepobs import analyzer
import tikzplotlib
import matplotlib as mpl
import codecs

# get the overall best performance of the MomentumOptimizer on the quadratic_deep testproblem
#no learning rate

setting = analyzer.shared_utils.create_setting_analyzer_ranking('./results/fmnist_2c2d/PreconditionedSGD',)[0]
mean = setting.aggregate["valid_accuracies"]['mean'][-1]
std = setting.aggregate["valid_accuracies"]['std'][-1]


# analyzer.plot_hyperparameter_sensitivity("./results/cifar10_3c3d/Preconditioner", hyperparam = 'lr')
fig, axes = analyzer.plot_hyperparameter_sensitivity("./results/mnist_vae/Preconditioner",
                                    reference_path = "../baselines/mnist_vae/SGD",
                                                    hyperparam = 'lr',
                                                    plot_std = False)



axes.set_xscale("log")
axes.set_ylabel("Accuracy")
axes.set_xlabel("Learning Rate")
axes.get_lines()[0].set_marker("x")
axes.get_lines()[1].set_marker("x")
axes.get_lines()[1].set_alpha(0.5)


axes.plot( [1e-5, 100], [mean, mean], linewidth = 3, linestyle = "dashed", color = "grey")

axes.legend(["PreconditionedSGD", "SGD", "Constructed Learning Rate"])
fig.canvas.draw()


code = tikzplotlib.get_tikz_code(figure = fig,
                                 figurewidth = "\\figurewidth + 1cm",
                                 figureheight = "5cm",
                                 extra_axis_parameters = ["tick pos=left",
             "legend style={font=\\footnotesize, at={(0.5 ,0)}, yshift=-1.5cm,anchor=north,nodes=right}"],
                                 extra_tikzpicture_parameters = [],
                                 )#strict = True)

#catch missed underscores & save
code = code.replace("\_", "_").replace("_", "\_")

file = codecs.open("../../thesis/images/exp_lr_sens.pgf", "w", 'utf-8')
file.write(code)
file.close()
