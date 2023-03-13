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
                                        Delta float NOT NULL,
                                        eps_0 float NOT NULL,
                                        d_eps float NOT NULL, 
                                        Gamma float NOT NULL,
                                        KbT float NOT NULL,
                                        omega float NOT NULL,
                                        ncycles integer,
                                        times array,
                                        steady_mask array,
                                        Relax boolean, 
                                        rho_0 array,
                                        dt float,
                                        sim_date text
                                );"""

def get_hash(Delta, eps0, d_eps, Gamma, KbT, omega, Relax, dt):

    string = f'D:{Delta:.12e};E0:{eps0:.12e};DE:{d_eps:.12e};GA:{Gamma:.12e};KT:{KbT:.12e};OM:{omega:.12e};DT:{dt:.12e};RE:{Relax:d}'
    
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

def create_simulation(conn, simulation):
    """
    Create a new project into the simulations table
    :param conn:
    :param project:
    :return: id
    """

    while True:
        try:
            sql = ''' INSERT INTO simulations (Probability, param_hash, Delta, eps_0, d_eps, Gamma, KbT, 
                                                omega, ncycles, times, steady_mask, Relax, rho_0, dt,sim_date)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
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

def save_simulation(conn, Gn, eps0, Delta, d_eps, omega, times, dt, rho_0, Relax , Gamma_c , KbT):
    
    date = get_today()

    hash_string = get_hash(Delta, eps0, d_eps, Gamma_c, KbT, omega, Relax, dt)

    ncycles = int(np.ceil(omega*times[-1]/(2*np.pi)))

    if Relax and Gamma_c:
        cycles_transient = np.ceil( omega/ Gamma_c )
        n_transient = min(np.ceil ( cycles_transient * 2*np.pi / (omega*dt) ), len(times))
        steady_mask = times >= n_transient*dt
    else:
        steady_mask = np.array([True]*len(times))

    sim = (Gn, hash_string, Delta, eps0, d_eps, Gamma_c, KbT, omega, ncycles, times, steady_mask, Relax, np.array(rho_0), dt,date)

    create_simulation(conn, sim)

def check_simulation(conn, eps0, Delta, d_eps, omega, times, dt, rho_0, Relax , Gamma_c , KbT):
    
    hash_string = get_hash(Delta, eps0, d_eps, Gamma_c, KbT, omega, Relax, dt)

    rows = select_from_hash(conn, hash_string)

    for P, rho_0_found in rows:
        if np.allclose(rho_0_found, rho_0):
            return P
    
    return None

def init_db(path):
    with Connection(path) as conn:
        create_table(conn, sql_create_sims_table)