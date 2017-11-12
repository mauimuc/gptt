.DEFAULT_GOAL := formulas.pdf

formulas.pdf: formulas.tex fig_path_coverage.pgf fig_correlation.pgf fig_example.pgf
	pdflatex formulas

def_example.tex ./dat/example.hdf5: ./src/example.py ./src/gptt.py
	cd src; python example.py

fig_example.pgf: ./src/fig_posterior.py ./dat/example.hdf5
	cd src; python fig_posterior.py

fig_path_coverage.pgf: ./src/fig_path_coverage.py ./src/example.py
	cd src; python fig_path_coverage.py

fig_correlation.pgf: ./src/fig_correlation.py ./src/example.py
	cd src; python fig_correlation.py

animation.mp4 animation.png: ./src/example.py ./src/animation.py
	cd src; python animation.py

clean:
	rm -f *.aux *.log *.out *.pgf
