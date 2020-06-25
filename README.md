# My B.Sc. Thesis
[📜 pdf: Investigating Probabilistic Preconditioning on Artificial Neural Networks](thesis/thesis_probprec.pdf)

[💻 pdf: Presentation](presentation/talk.pdf)

This repository contains all sources and experimental data for my B.Sc. thesis, in which I evaluate the performance of a probabilistic preconditioning algorithm on neural networks using the benchmarking suite DeepOBS.

I wrote this thesis in 2019 in [Philipp Hennig](https://twitter.com/PhilippHennig5)'s research group [Methods of Machine Learning](https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/computer-science/lehrstuehle/methods-of-machine-learning/start/) at the University of Tübingen, Germany.

It is based on these projects, using DeepOBS to evaluate a new optimization algorithm:
* DeepOBS by Frank Schneider, specifically Aaron Bahde's pytorch implementation (https://github.com/abahde/DeepOBS)
* A NeurIPS paper by my supervisor Filip De Roos and our PI Philipp Hennig.
(a related project: https://github.com/fderoos/probabilistic_hessian)

## Open Data
In `code`, find all the code, most importantly the `Precoditioner` class in `code/probprec.py`. The experiment folders contain all files necessary to replicate an experiment and generate the corresponding figure. And the sources for the written thesis and the defense presentation are in their respective folders

## Technical description
The technical setup is decribed in full in the thesis, but here's a quick overview:

1. Experiments were run using pytorch and DeepOBS in a Singularity container on the TCML cluster
provided by the University of Tübingen.
2. The presentation is based on the internal LaTeX template of the MoML chair and is meant to be viewed on two screens simultaneously.
3. The thesis is based on the [english language LaTeX template](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/integrative-transkriptomik/theses/) as provided by Prof. Kay Nieselt.
