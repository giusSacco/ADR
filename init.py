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
save_c_avg = cfg.getboolean('saving', 'save_c_avg')
save_c_constwind_avg = cfg.getboolean('saving', 'save_c_constwind_avg')
gifname = cfg.get('saving', 'gifname').format(alpha_x0,d_alpha_x,N_period)
fname_c_avg = cfg.get('saving','fname_c_avg')
fname_c_constwind_avg = cfg.get('saving','fname_c_constwind_avg')
show_3Dplot = cfg.getboolean('plotting params', 'show_3Dplot')
save_array_dir = cfg.get('saving', 'save_array_dir')
second_plot_name = cfg.get('saving', 'second_plot_name').format(alpha_x0,d_alpha_x,N_period)
rand_seed = cfg.getint('simulation params', 'rand_seed')
make_first_plot = cfg.getboolean('plotting params', 'make_first_plot')
N_t_stationary_min = cfg.getint('simulation params', 'N_t_stationary_min')
stationary_start = cfg.getboolean('simulation params', 'stationary_start')
stat_pops_dir = cfg.get('saving','stat_pops_dir')
stationary_c_fname = os.path.join(stat_pops_dir,cfg.get('saving','stationary_c_fname').format(alpha_x0,d_alpha_x,N_period))
stationary_c_constwind_fname = os.path.join(stat_pops_dir,cfg.get('saving','stationary_c_constwind_fname').format(alpha_x0))

ADR_keywords = ['Nx','Ny','Nt','N_period','dt','alpha_x0','d_alpha_x','delta','b','a_chem_m','a_chem_range','c_m','c_range',\
                'savefig_dir','dnk1','save_c_avg','save_c_constwind_avg','gifname','fname_c_avg','fname_c_constwind_avg',\
                'show_3Dplot','save_array_dir','second_plot_name','rand_seed','make_first_plot','N_t_stationary_min',\
                'stationary_start','stat_pops_dir','stationary_c_fname','stationary_c_constwind_fname']

ADR_params_dict = {}
for key in ADR_keywords:
    if key not in locals():
        raise ValueError('Key {} not found in ADR_keywords'.format(key))
    ADR_params_dict[key] = locals()[key]