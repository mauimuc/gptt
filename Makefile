.DEFAULT_GOAL := formulas.pdf

formulas.pdf: formulas.tex fig_path_coverage.pgf fig_correlation.pgf fig_example.pgf
	pdflatex formulas

fig_example.pgf def_example.pgf: ./src/example.py ./src/gptt.py
	cd src; python example.py

fig_path_coverage.pgf fig_reference_model.pgf: ./src/fig_reference_model.py ./src/example.py
	cd src; python fig_reference_model.py

fig_discretization.pgf fig_kernel.pgf fig_correlation.pgf: ./src/fig_correlation.py ./src/example.py
	cd src; python fig_correlation.py

presentation.pdf: presentation.tex fig_reference_model.pgf fig_path_coverage.pgf fig_kernel.pgf fig_correlation.pgf fig_discretization.pgf animation.mp4 animation_pst.png
	pdflatex presentation

animation.mp4 animation_pri.png animation_pst.png: ./src/example.py ./src/animation.py
	cd src; python animation.py

clean:
	rm -f *.aux *.log *.out *.pgf *.png
