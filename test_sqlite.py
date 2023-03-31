from sqlite_handler import select_from_param, Connection, select_all_sims

# Path to the database
path_to_db = './Simulations.db'

# This example is for the case where you want to select all the simulations with a certain parameter value
with Connection(path_to_db) as conn:
    r =  select_all_sims(conn)
    print(len(r))

# This example is for the case where you want to select all the simulations with a certain parameter value
parameters_to_retreive = ['c_avg','N_t_stationary','c_constwind_avg' ]

parameter_to_query = 'Nt'
value_to_query = 300

with Connection(path_to_db) as conn:
    r =  select_from_param(conn, parameter_to_query, value_to_query, parameters_to_retreive)
    # print(r)

