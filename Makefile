.DEFAULT_GOAL := formulas.pdf

formulas.pdf: formulas.tex fig_path_coverage.pgf fig_correlation.pgf fig_example.pgf
	pdflatex formulas

fig_example.pgf: ./src/example.py
	cd src; python example.py

fig_path_coverage.pgf: ./src/fig_path_coverage.py ./src/plotting.py
	cd src; python fig_path_coverage.py

fig_correlation.pgf: ./src/fig_correlation.py ./src/plotting.py
	cd src; python fig_correlation.py

