import sqlite3
import numpy as np
import io
from hashlib import sha256
from contextlib import contextmanager
from datetime import datetime
from time import sleep

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
                                        stationary_start integer NOT NULL,
                                        N_t_stationaty_min integer NOT NULL,
                                        do_const_wind boolean NOT NULL,
                                        do_FT boolean NOT NULL,
                                        savefig_dir text NOT NULL,
                                        dnk1 integer NOT NULL,
                                        show_3D_plot boolean NOT NULL,
                                        make_first_plot boolean NOT NULL,
                                        stat_pops_dir text NOT NULL,
                                        stationary_c_fname text NOT NULL,
                                        stationary_c_constwind_fname text NOT NULL,
                                        save_array_dir text NOT NULL,
                                        save_c_avg boolean NOT NULL,
                                        fname_c_avg text NOT NULL,
                                        save_c_constwind_avg boolean NOT NULL,
                                        fname_c_constwind_avg text NOT NULL,
                                        gifname text NOT NULL,
                                        second_plot_name text NOT NULL,
                                        sim_date text 
                                );"""


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

def create_simulation(conn, simulation):
    """
    Create a new project into the simulations table
    :param conn:
    :param project:
    :return: id
    """

    while True:
        try:
            sql = ''' INSERT INTO simulations (Nx, Ny, Nt, N_period, alpha_x0, d_alpha_x, delta, dt, b, a_chem_m, a_chem_range, c_m, c_range, rand_seed, stationary_start, N_t_stationaty_min, do_const_wind, do_FT, savefig_dir, dnk1, show_3D_plot, make_first_plot, stat_pops_dir, stationary_c_fname, stationary_c_constwind_fname, save_array_dir, save_c_avg, fname_c_avg, save_c_constwind_avg, fname_c_constwind_avg, gifname, second_plot_name, sim_date)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
            cur = conn.cursor()
            cur.execute(sql, simulation)
            conn.commit()
            return cur.lastrowid
        except Exception as e:
            # print(f'CREATE SIM {type(e).__name__} :  {e}')
            sleep(0.001)

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

def select_from_hash(conn, hash_string):

    """
    Query all rows in the simulation table
    :param conn: the Connection object
    :return:
    """

    cur = conn.cursor()
    cur.execute("SELECT Probability, rho_0 FROM simulations WHERE param_hash=?", (hash_string,))

    rows = cur.fetchall()

    return rows

@contextmanager
def Connection(name = ':memory:'):

    conn = create_connection(name)

    yield conn
    
    conn.close()

def save_simulation(conn, Nx, Ny, Nt, N_period, alpha_x0, d_alpha_x, delta, dt, b, a_chem_m, a_chem_range, c_m, c_range, rand_seed, stationary_start, N_t_stationaty_min, do_const_wind, do_FT, savefig_dir, dnk1, show_3D_plot, make_first_plot, stat_pops_dir, stationary_c_fname, stationary_c_constwind_fname, save_array_dir, save_c_avg, fname_c_avg, save_c_constwind_avg, fname_c_constwind_avg, gifname, second_plot_name):
    
    date = get_today()
    sim = (Nx, Ny, Nt, N_period, alpha_x0, d_alpha_x, delta, dt, b, a_chem_m, a_chem_range, c_m, c_range, rand_seed, stationary_start, N_t_stationaty_min, do_const_wind, do_FT, savefig_dir, dnk1, show_3D_plot, make_first_plot, stat_pops_dir, stationary_c_fname, stationary_c_constwind_fname, save_array_dir, save_c_avg, fname_c_avg, save_c_constwind_avg, fname_c_constwind_avg, gifname, second_plot_name, date)

    create_simulation(conn, sim)

def init_db(path):
    with Connection(path) as conn:
        create_table(conn, sql_create_sims_table)