.DEFAULT_GOAL := formulas.pdf

formulas.pdf: formulas.tex fig_path_coverage.pgf fig_correlation.pgf
	pdflatex formulas

fig_path_coverage.pgf: ./src/fig_path_coverage.py
	cd src; python fig_path_coverage.py

fig_correlation.pgf: ./src/fig_correlation.py
	cd src; python fig_correlation.py

