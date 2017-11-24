.DEFAULT_GOAL := all

all: formulas.pdf presentation.pdf

# Overwrite pseudo data only for a very good reason
./dat/pseudo_data.dat: ./src/reference.py
	cd src; python reference.py

./dat/all_at_once.hdf5: ./src/all_at_once.py ./src/gptt.py ./dat/pseudo_data.dat ./par/example.ini
	cd src; python all_at_once.py
fig_all_at_once.pgf: ./dat/all_at_once.hdf5 ./src/fig_posterior.py
	cd src; python fig_posterior.py ../dat/all_at_once.hdf5

formulas.pdf: formulas.tex fig_reference_model.pgf fig_path_coverage.pgf fig_discretization.pgf fig_correlation_pri.pgf fig_example.pgf fig_all_at_once.pgf fig_example_ascending.pgf fig_example_descending.pgf
	pdflatex formulas

def_example.tex ./dat/example.hdf5: ./src/example.py ./src/gptt.py ./dat/pseudo_data.dat ./par/example.ini
	cd src; python example.py ../par/example.ini
./dat/example_ascending.hdf5: ./src/example.py ./src/gptt.py ./dat/pseudo_data.dat ./par/example_ascending.ini
	cd src; python example.py ../par/example_ascending.ini
./dat/example_descending.hdf5: ./src/example.py ./src/gptt.py ./dat/pseudo_data.dat ./par/example_descending.ini
	cd src; python example.py ../par/example_descending.ini

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
	cd src; python fig_posterior.py ../dat/example.hdf5

fig_correlation_pst.pgf fig_kernel_pst.pgf: ./src/fig_correlation_pst.py ./dat/example.hdf5
	cd src; python fig_correlation_pst.py

.PHONY: misfit
misfit: ./src/fig_misfit.py ./dat/example.hdf5 ./dat/example_ascending.hdf5 ./dat/example_descending.hdf5 ./dat/all_at_once.hdf5
	cd src; python fig_misfit.py

presentation.pdf: presentation.tex fig_reference_model.pgf fig_path_coverage.pgf fig_kernel_pri.pgf fig_correlation_pri.pgf fig_discretization.pgf animation.avi animation_pst.png fig_correlation_pst.pgf fig_kernel_pst.pgf
	pdflatex presentation

animation.avi animation_pri.png animation_pst.png: ./dat/example.hdf5 ./src/animation.py
	cd src; python animation.py

fig_example_ascending.pgf: ./dat/example_ascending.hdf5 ./src/fig_posterior.py
	cd src; python fig_posterior.py ../dat/example_ascending.hdf5

fig_example_descending.pgf: ./dat/example_descending.hdf5 ./src/fig_posterior.py
	cd src; python fig_posterior.py ../dat/example_descending.hdf5


clean:
	rm -f *.aux *.log *.out *.pgf *.png ./dat/*.hdf5 ./src/*.pyc
