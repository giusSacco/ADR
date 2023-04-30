import sqlite3
import numpy as np
import io
from hashlib import sha256
from contextlib import contextmanager
from datetime import datetime
from time import sleep
from typing import List, Tuple, Union
from hashlib import sha256

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, np.array(arr))
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle = False)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

sql_create_sims_table = """ CREATE TABLE IF NOT EXISTS simulations (
                                        sim_id integer PRIMARY KEY,
                                        Nx integer NOT NULL,
                                        Ny integer NOT NULL,
                                        Nt integer NOT NULL,
                                        N_period integer NOT NULL,
                                        alpha_x0 real NOT NULL,
                                        d_alpha_x real NOT NULL,
                                        delta real NOT NULL,
                                        dt real NOT NULL,
                                        b real NOT NULL,
                                        a_chem_m real NOT NULL,
                                        a_chem_range real NOT NULL,
                                        c_m real NOT NULL,
                                        c_range real NOT NULL,
                                        rand_seed integer NOT NULL,
                                        N_t_stationary_min integer NOT NULL,
                                        do_const_wind boolean NOT NULL,
                                        do_FT boolean NOT NULL,
                                        savefig_dir text NOT NULL,
                                        dnk1 integer NOT NULL,
                                        show_3Dplot boolean NOT NULL,
                                        make_first_plot boolean NOT NULL,
                                        gifname text NOT NULL,
                                        second_plot_name text NOT NULL,
                                        c_avg array NOT NULL,
                                        N_t_stationary integer NOT NULL,
                                        c_constwind_avg array,
                                        c_full_frame_one_per array,
                                        elapsed_time real,
                                        param_hash text NOT NULL,
                                        sim_date text 
                                );"""


params = ['Nx', 'Ny', 'Nt', 'N_period', 'alpha_x0', 'd_alpha_x', 
          'delta', 'dt', 'b', 'a_chem_m', 'a_chem_range', 'c_m', 'c_range', 
          'rand_seed', 'N_t_stationary_min', 'do_const_wind', 'do_FT', 
          'savefig_dir', 'dnk1', 'show_3Dplot', 'make_first_plot', 
          'gifname', 'second_plot_name', 'c_avg', 'N_t_stationary', 
          'c_constwind_avg', 'c_full_frame_one_per', 'elapsed_time', 'param_hash', 'sim_date']

params_in_hash = ['Nx', 'Ny', 'Nt', 'N_period', 'alpha_x0', 'd_alpha_x', 'delta', 'dt', 'b', 'a_chem_m', 'a_chem_range', 'c_m', 'c_range', 'rand_seed', 'N_t_stationary_min']

def get_hash(Nx, Ny, Nt, N_period, alpha_x0, d_alpha_x, 
             delta, dt, b, a_chem_m, a_chem_range, 
             c_m, c_range, rand_seed, 
             N_t_stationary_min):

    string = ''

    for param in params_in_hash:
        string += f'{param}:{eval(param)}:'
    
    return hash_string(string)

def hash_string(string):
    return sha256(string.encode('utf-8')).hexdigest()

def get_today():
    return datetime.today().strftime('%Y-%m-%d %H:%M')

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, timeout = 10)
        return conn
    except Exception as e:
        print(f'CONNECTION {type(e).__name__} :  {e}')

    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Exception as e:
        print(f'CREATE TABLE {type(e).__name__} :  {e}')

def create_simulation(conn, simulation, maxtries = 100):
    """
    Create a new project into the simulations table
    :param conn:
    :param project:
    :return: id
    """
    i = 0
    while True:
        i+=1
        try:
            sql = ''' INSERT INTO simulations ('''
            sql += ','.join(params)
            sql += ''') '''
            sql +=  'VALUES('+ ','.join(['?']*len(params)) +')'
            cur = conn.cursor()
            cur.execute(sql, simulation)
            conn.commit()
            return cur.lastrowid
        except Exception as e:
            if i > maxtries:
                raise e
            # print(f'CREATE SIM {type(e).__name__} :  {e}')
            sleep(0.01)

def select_all_sims(conn):

    """
    Query all rows in the simulation table
    :param conn: the Connection object
    :return:
    """

    cur = conn.cursor()
    cur.execute("SELECT * FROM simulations")

    rows = cur.fetchall()

    return rows

def select_from_multiple_params(conn, params_to_query: Sequence[str], values :  Sequence[Union[str, int, float]],
                                 params_to_return : List[str] = ['c_avg','N_t_stationary','c_constwind_avg' ]):
    
    """
    Query rows in the simulation table
    :param conn: the Connection object
    :param params_to_query: the parameters to query
    :param values: the values of the parameters
    :param params_to_return: the parameters to return
    """

    for p in params_to_return:
        if p not in params:
            raise ValueError(f'Parameter {p} not in {params}')

    cur = conn.cursor()
    cur.execute(f"SELECT {','.join(params_to_return)} FROM simulations WHERE {'AND '.join([f'{p}=? ' for p in params_to_query])}", tuple(values))

    rows = cur.fetchall()

    return rows

def select_from_param(conn, param : str, value : Union[str, int, float], 
                    params_to_return : List[str] = ['c_avg','N_t_stationary','c_constwind_avg' ]):

    """
    Query rows in the simulation table
    :param conn: the Connection object
    :param param: the parameter to query
    :param value: the value of the parameter
    :param params_to_return: the parameters to return
    """

    for p in params_to_return:
        if p not in params:
            raise ValueError(f'Parameter {p} not in {params}')

    cur = conn.cursor()
    cur.execute(f"SELECT {','.join(params_to_return)} FROM simulations WHERE {param}=?", (value,))

    rows = cur.fetchall()

    return rows

@contextmanager
def Connection(name = ':memory:'):

    conn = create_connection(name)

    yield conn
    
    conn.close()

def save_simulation(conn, Nx, Ny, Nt, N_period, 
                    alpha_x0, d_alpha_x, delta, 
                    dt, b, a_chem_m, a_chem_range, 
                    c_m, c_range, 
                    rand_seed, N_t_stationary_min, 
                    do_const_wind, do_FT, savefig_dir, 
                    dnk1, show_3Dplot, make_first_plot, 
                    gifname, second_plot_name, 
                    c_avg, N_t_stationary, c_constwind_avg, 
                    c_full_frame_one_per, elapsed_time, maxtries = 100):
    
    date = get_today()

    param_hash = get_hash(Nx, Ny, Nt, N_period, alpha_x0, d_alpha_x, delta, dt, b, a_chem_m, a_chem_range, c_m, c_range, rand_seed, N_t_stationary_min)
    if select_from_param(conn, 'param_hash', param_hash, params_to_return = ['param_hash']):
        return
    else:
        sim = (Nx, Ny, Nt, N_period, alpha_x0, d_alpha_x, delta, dt, b, a_chem_m, a_chem_range, c_m, c_range, rand_seed, N_t_stationary_min, do_const_wind, do_FT, savefig_dir, dnk1, show_3Dplot, make_first_plot, gifname, second_plot_name, c_avg, N_t_stationary, c_constwind_avg, c_full_frame_one_per, elapsed_time, param_hash, date)
        create_simulation(conn, sim, maxtries=maxtries)

def init_db(path):
    with Connection(path) as conn:
        create_table(conn, sql_create_sims_table)