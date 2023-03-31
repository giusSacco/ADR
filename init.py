from configparser import ConfigParser
import os

# Parsing all parameters and general infos from ini file 'parameters.ini'
# For details on the following definitions, check the ini file.
cfg = ConfigParser()
cfg.read('parameters.ini')

Nx = cfg.getint('simulation params', 'Nx')
Ny = cfg.getint('simulation params', 'Ny')
Nt = cfg.getint('simulation params', 'Nt')
N_period = cfg.getint('simulation params', 'N_period')
dt = cfg.getfloat('simulation params', 'dt')
alpha_x0 = cfg.getfloat('simulation params', 'alpha_x0')
d_alpha_x = cfg.getfloat('simulation params', 'd_alpha_x')
delta = cfg.getfloat('simulation params', 'delta')
b = cfg.getfloat('simulation params', 'b')
a_chem_m = cfg.getfloat('simulation params', 'a_chem_m')
a_chem_range = cfg.getfloat('simulation params', 'a_chem_range')
c_m = cfg.getfloat('simulation params', 'c_m')
c_range = cfg.getfloat('simulation params', 'c_range')
savefig_dir = cfg.get('files and directories', 'savefig_dir').format(alpha_x0,d_alpha_x,N_period)
dnk1 = cfg.getint('plotting params', 'dnk1')

gifname = cfg.get('saving', 'gifname').format(alpha_x0,d_alpha_x,N_period)

show_3Dplot = cfg.getboolean('plotting params', 'show_3Dplot')

second_plot_name = cfg.get('saving', 'second_plot_name').format(alpha_x0,d_alpha_x,N_period)
rand_seed = cfg.getint('simulation params', 'rand_seed')
make_first_plot = cfg.getboolean('plotting params', 'make_first_plot')
N_t_stationary_min = cfg.getint('simulation params', 'N_t_stationary_min')

do_FT = cfg.getboolean('simulation params', 'do_FT')
do_const_wind = cfg.getboolean('simulation params', 'do_const_wind')
if do_FT is True and do_const_wind is False:
    raise ValueError('do_FT is True but do_const_wind is False. This is not allowed. Plaease set do_const_wind to True if you wish to see FT results.')

ADR_keywords = ['Nx','Ny','Nt','N_period','dt','alpha_x0','d_alpha_x',\
                'delta','b','a_chem_m','a_chem_range','c_m','c_range',\
                'savefig_dir','dnk1','gifname',\
                'show_3Dplot','second_plot_name','rand_seed',   
                'make_first_plot','N_t_stationary_min',\
                'do_FT', 'do_const_wind']

ADR_params_dict = {}
for key in ADR_keywords:
    if key not in locals():
        raise ValueError('Key {} not found in ADR_keywords'.format(key))
    ADR_params_dict[key] = locals()[key]