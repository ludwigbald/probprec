ALL_FIGURE_NAMES=$(shell cat talk.figlist)
ALL_FIGURES=$(ALL_FIGURE_NAMES:%=%.pdf)

allimages: $(ALL_FIGURES)
	@echo All images exist now. Use make -B to re-generate them.

FORCEREMAKE:

include $(ALL_FIGURE_NAMES:%=%.dep)

%.dep:
	mkdir -p "$(dir $@)"
	touch "$@" # will be filled later.

fig/external/talk-figure0.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "fig/external/talk-figure0" "\def\tikzexternalrealjob{talk}\input{talk}"

fig/external/talk-figure0.pdf: fig/external/talk-figure0.md5
fig/external/talk-figure1.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "fig/external/talk-figure1" "\def\tikzexternalrealjob{talk}\input{talk}"

fig/external/talk-figure1.pdf: fig/external/talk-figure1.md5
fig/external/talk-figure2.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "fig/external/talk-figure2" "\def\tikzexternalrealjob{talk}\input{talk}"

fig/external/talk-figure2.pdf: fig/external/talk-figure2.md5
fig/external/talk-figure3.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "fig/external/talk-figure3" "\def\tikzexternalrealjob{talk}\input{talk}"

fig/external/talk-figure3.pdf: fig/external/talk-figure3.md5
fig/external/talk-figure4.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "fig/external/talk-figure4" "\def\tikzexternalrealjob{talk}\input{talk}"

fig/external/talk-figure4.pdf: fig/external/talk-figure4.md5
fig/external/talk-figure5.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "fig/external/talk-figure5" "\def\tikzexternalrealjob{talk}\input{talk}"

fig/external/talk-figure5.pdf: fig/external/talk-figure5.md5
fig/external/talk-figure6.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "fig/external/talk-figure6" "\def\tikzexternalrealjob{talk}\input{talk}"

fig/external/talk-figure6.pdf: fig/external/talk-figure6.md5
fig/external/talk-figure7.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "fig/external/talk-figure7" "\def\tikzexternalrealjob{talk}\input{talk}"

fig/external/talk-figure7.pdf: fig/external/talk-figure7.md5
