.DEFAULT_GOAL := formulas.pdf

formulas.pdf: formulas.tex fig_path_coverage.pgf fig_correlation.pgf fig_example_mu.pgf fig_example_sd.pgf
	pdflatex formulas

fig_example_mu.pgf fig_example_sd.pgf: ./src/example.py ./src/gptt.py
	cd src; python example.py

fig_path_coverage.pgf: ./src/fig_path_coverage.py ./src/plotting.py
	cd src; python fig_path_coverage.py

fig_correlation.pgf: ./src/fig_correlation.py ./src/plotting.py
	cd src; python fig_correlation.py

example.mp4: ./src/example.py ./src/animation.py
	cd src; python animation.py

clean:
	rm -f *.aux *.log *.out *.pgf
