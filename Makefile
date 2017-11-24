.DEFAULT_GOAL := all


all: formulas.pdf presentation.pdf

# Overwrite pseudo data only for a very good reason
./dat/pseudo_data.dat: ./src/reference.py
	cd src; python reference.py


# TODO Link data and parameter file
./dat/%.hdf5: ./par/%.ini ./src/example.py ./src/gptt.py ./dat/pseudo_data.dat
	cd src; python example.py ../$<

./dat/all_at_once.hdf5: ./src/all_at_once.py ./src/gptt.py ./dat/pseudo_data.dat ./par/example.ini
	cd src; python all_at_once.py

formulas.pdf: formulas.tex fig_reference_model.pgf fig_path_coverage.pgf fig_discretization.pgf fig_correlation_pri.pgf fig_example.pgf fig_all_at_once.pgf fig_example_ascending.pgf fig_example_descending.pgf
	pdflatex formulas

def_example.tex: ./src/def_example.py ./par/example.ini
	cd src; python def_example.py ../par/example.ini

fig_reference_model.pgf: ./src/fig_reference_model.py ./src/plotting.py ./dat/pseudo_data.dat
	cd src; python fig_reference_model.py

fig_discretization.pgf: ./src/fig_discretization.py ./src/plotting.py ./dat/pseudo_data.dat
	cd src; python fig_discretization.py

fig_path_coverage.pgf: ./src/fig_path_coverage.py ./src/plotting.py ./dat/pseudo_data.dat
	cd src; python fig_path_coverage.py

fig_kernel_pri.pgf: ./src/fig_kernel_pri.py ./src/plotting.py ./par/example.ini
	cd src; python fig_kernel_pri.py

fig_correlation_pri.pgf: ./src/fig_correlation_pri.py ./src/plotting.py ./par/example.ini
	cd src; python fig_correlation_pri.py

fig_%.pgf: ./dat/%.hdf5 ./src/fig_posterior.py
	cd src; python fig_posterior.py ../$<

fig_correlation_pst.pgf fig_kernel_pst.pgf: ./src/fig_correlation_pst.py ./dat/example.hdf5
	cd src; python fig_correlation_pst.py

.PHONY: misfit
misfit: ./src/fig_misfit.py ./dat/example.hdf5 ./dat/example_ascending.hdf5 ./dat/example_descending.hdf5 ./dat/all_at_once.hdf5
	cd src; python fig_misfit.py

presentation.pdf: presentation.tex fig_reference_model.pgf fig_path_coverage.pgf fig_kernel_pri.pgf fig_correlation_pri.pgf fig_discretization.pgf animation.avi animation_pst.png fig_correlation_pst.pgf fig_kernel_pst.pgf
	pdflatex presentation

animation.avi animation_pri.png animation_pst.png: ./dat/example.hdf5 ./src/animation.py
	cd src; python animation.py

clean:
	rm -f *.aux *.log *.out *.pgf *.png ./dat/*.hdf5 ./src/*.pyc
