.DEFAULT_GOAL := all

all: formulas.pdf presentation.pdf

# Overwrite pseudo data only for a very good reason
./dat/pseudo_data.dat: ./src/reference.py
	cd src; python reference.py

./dat/all_at_once.hdf5: ./src/all_at_once.py ./dat/pseudo_data.dat ./src/example.py
	cd src; python all_at_once.py
fig_all_at_once.pgf: ./dat/all_at_once.hdf5 ./src/fig_posterior.py
	cd src; python fig_posterior.py ../dat/all_at_once.hdf5

formulas.pdf: formulas.tex fig_reference_model.pgf fig_path_coverage.pgf fig_discretization.pgf fig_correlation_pri.pgf fig_example.pgf fig_all_at_once.pgf fig_succession_sorted.pgf fig_succession_reverse.pgf
	pdflatex formulas

def_example.tex ./dat/example.hdf5: ./src/example.py ./src/gptt.py ./dat/pseudo_data.dat ./src/parameter.ini
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
	cd src; python fig_posterior.py ../dat/example.hdf5

fig_correlation_pst.pgf fig_kernel_pst.pgf: ./src/fig_correlation_pst.py ./dat/example.hdf5
	cd src; python fig_correlation_pst.py

#fig_misfit.pgf: ./src/fig_misfit.py ./dat/example.hdf5
#	cd src; python fig_misfit.py

presentation.pdf: presentation.tex fig_reference_model.pgf fig_path_coverage.pgf fig_kernel_pri.pgf fig_correlation_pri.pgf fig_discretization.pgf animation.avi animation_pst.png fig_correlation_pst.pgf fig_kernel_pst.pgf
	pdflatex presentation

animation.avi animation_pri.png animation_pst.png: ./dat/example.hdf5 ./src/animation.py
	cd src; python animation.py

./dat/succession_sorted.hdf5 ./dat/succession_reverse.hdf5: ./src/succession_sorted.py ./dat/pseudo_data.dat ./src/example.py
	cd src; python succession_sorted.py

fig_succession_sorted.pgf: ./dat/succession_sorted.hdf5 ./src/fig_posterior.py
	cd src; python fig_posterior.py ../dat/succession_sorted.hdf5

fig_succession_reverse.pgf: ./dat/succession_reverse.hdf5 ./src/fig_posterior.py
	cd src; python fig_posterior.py ../dat/succession_reverse.hdf5


clean:
	rm -f *.aux *.log *.out *.pgf *.png ./dat/*.hdf5 ./src/*.pyc
