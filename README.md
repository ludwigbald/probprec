# probprec
All sources for my BSc thesis on evaluating probabilistic preconditioning using DeepOBS.

I am working at the chair of Philipp Hennig (Methods of Machine Learning) at the University of Tübingen, Germany.

Based on these projects:

* DeepOBS by Frank Schneider and Aaron Bahde (https://github.com/abahde/DeepOBS)
* A NeurIPS paper by my supervisor Filip De Roos and our PI Philipp Hennig.
(a related project: https://github.com/fderoos/probabilistic_hessian)

## Structure of this repository
I tried to create a comprehensive structure. In `code`, find all the code, most notably the `Precoditioner` class in `code/probprec.py`. Also find the experiment folders, which contain all files necessary to replicate an experiment and generate the corresponding figure.

Thesis sources are in `thesis`, presentation sources are in `presentation`.


## Technical description
Most of the technical parts are decribed in detail in the thesis, but here's a quick overview:

1. Experiments were run using pytorch and DeepOBS in a Singularity container on the TCML cluster
provided by the University of Tübingen.
2. The presentation is based on the internal template of the MoML chair.
3. The thesis is based on the english language template provided by Kay Nieselt here: https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/integrative-transkriptomik/theses/
