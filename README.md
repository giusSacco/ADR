# ADR

"ADR.py" takes as input the parameters written in "parameters.ini" through "init.py".
It than creates through the functions saved in "custom_plots.py" the gif (also single figures must be saved) of the time evolution, the 3D plot and the colormap in the k_x, t plane.
It also prints in two files the arrays of the average population and average nowind population for each time step. Each of those can later be plotted by "pop_plotter.py" (beware of how you name the files since pop_plotter uses regular expressions).