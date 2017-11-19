.DEFAULT_GOAL := formulas.pdf

./dat/pseudo_data.dat: ./src/reference.py
	cd src; python reference.py

formulas.pdf: formulas.tex fig_reference_model.pgf fig_path_coverage.pgf fig_discretization.pgf fig_correlation_pri.pgf fig_example.pgf
	pdflatex formulas

def_example.tex ./dat/example.hdf5: ./src/example.py ./src/gptt.py
	cd src; python example.py

fig_reference_model.pgf: ./src/fig_reference_model.py ./src/plotting.py ./dat/pseudo_data.dat
	cd src; python fig_reference_model.py

fig_discretization.pgf: ./src/fig_discretization.py ./src/plotting.py ./dat/pseudo_data.dat
	cd src; python fig_discretization.py

fig_path_coverage.pgf: ./src/fig_path_coverage.py ./src/plotting.py ./dat/pseudo_data.dat
	cd src; python fig_path_coverage.py

fig_kernel_pri.pgf: ./src/fig_kernel_pri.py ./src/example.py ./src/plotting.py
	cd src; python fig_kernel_pri.py

fig_correlation_pri.pgf: ./src/fig_correlation_pri.py ./src/example.py ./src/plotting.py
	cd src; python fig_correlation_pri.py

fig_example.pgf: ./src/fig_posterior.py ./dat/example.hdf5
	cd src; python fig_posterior.py

fig_correlation_pst.pgf fig_kernel_pst.pgf: ./src/fig_correlation_pst.py ./dat/example.hdf5
	cd src; python fig_correlation_pst.py

fig_misfit.pgf: ./src/fig_misfit.py ./dat/example.hdf5
	cd src; python fig_misfit.py

presentation.pdf: presentation.tex fig_reference_model.pgf fig_path_coverage.pgf fig_kernel_pri.pgf fig_correlation_pri.pgf fig_discretization.pgf animation.mp4 animation_pst.png fig_correlation_pst.pgf fig_kernel_pst.pgf fig_misfit.pgf
	pdflatex presentation

animation.mp4 animation_pri.png animation_pst.png: ./dat/example.hdf5 ./src/animation.py
	cd src; python animation.py

clean:
	rm -f *.aux *.log *.out *.pgf *.png ./dat/*.hdf5
