from configparser import ConfigParser
import numpy as np

# Parsing all parameters and general infos from ini file 'parameters.ini'
# For details on the following definitions, check the ini file.
cfg = ConfigParser()
cfg.read('parameters.ini')

Nx = cfg.getint('simulation params', 'Nx')
Ny = cfg.getint('simulation params', 'Ny')
#Nt = cfg.getint('simulation params', 'Nt')
N_periods = np.geomspace(1, 10000,100, endpoint=True)
Nts=[]
for i in range(len(N_periods)):
    Nts.append(int(max(2*np.ceil(N_periods[i]), 1000+np.ceil(N_periods[i])))) 
#N_period = cfg.getint('simulation params', 'N_period')
dt = cfg.getfloat('simulation params', 'dt')
alpha_x0 = cfg.getfloat('simulation params', 'alpha_x0')
d_alpha_x = cfg.getfloat('simulation params', 'd_alpha_x')
delta = cfg.getfloat('simulation params', 'delta')
b = cfg.getfloat('simulation params', 'b')
a_chem_m = cfg.getfloat('simulation params', 'a_chem_m')
a_chem_range = cfg.getfloat('simulation params', 'a_chem_range')
c_m = cfg.getfloat('simulation params', 'c_m')
c_range = cfg.getfloat('simulation params', 'c_range')
#savefig_dir = cfg.get('files and directories', 'savefig_dir').format(alpha_x0,d_alpha_x,N_period)
savefig_dir = cfg.get('files and directories', 'savefig_dir').format(alpha_x0,d_alpha_x)
dnk1 = cfg.getint('plotting params', 'dnk1')
save_c_avg = cfg.getboolean('saving', 'save_c_avg')
save_c_nowind_avg = cfg.getboolean('saving', 'save_c_nowind_avg')
#gifname = cfg.get('saving', 'gifname').format(alpha_x0,d_alpha_x,N_period)
gifname = cfg.get('saving', 'gifname').format(alpha_x0,d_alpha_x)
fname_c_avg = cfg.get('saving','fname_c_avg')
fname_c_nowind_avg = cfg.get('saving','fname_c_nowind_avg')
show_3Dplot = cfg.getboolean('plotting params', 'show_3Dplot')
save_array_dir = cfg.get('saving', 'save_array_dir')
#second_plot_name = cfg.get('saving', 'second_plot_name').format(alpha_x0,d_alpha_x,N_period)
second_plot_name = cfg.get('saving', 'second_plot_name').format(alpha_x0,d_alpha_x)
rand_seed = cfg.getint('simulation params', 'rand_seed')
make_first_plot = cfg.getboolean('plotting params', 'make_first_plot')