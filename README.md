# ADR

"ADR.py" takes as input the parameters written in "parameters.ini" through "init.py".
In order to run different simulations with spanned parameters, one can write the latter by running "ADR_multiprocessing.py" and explicitely state the parameters there.
Input parameters and outputs of the simulations are saved in Simulations.db (handled by sqlite_handler.py).
If asked, ADR.py creates through the functions saved in "custom_plots.py" variouses plots.
