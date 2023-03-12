import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read('parameters.ini')

#Nt = 1000
Nt = cfg.getint('simulation params', 'Nt')
alpha_x0 = 0.15
d_alpha_x = 0.15
a_chem_m = cfg.getfloat('simulation params', 'a_chem_m')
save_array_dir = cfg.get('saving', 'save_array_dir')


fig = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])

ax1 = plt.subplot(gs[0])
ax1.set_xlabel('$n_t$')
ax1.set_ylabel(r'$\frac{<c(t)>-<c_0(t)>}{<c(0)>}$', size = 13)
ax1.set_xlim(0,Nt)

ax2 = plt.subplot(gs[1])
ax2.set_xlabel('$n_t$')
ax2.set_ylabel(r'$\alpha$')
ax2.set_xlim(0,Nt)
ax1.grid()
ax2.grid()

pattern = re.compile( r'''omega_(\d+)\.npy''')
FILES = tuple([ file_ for file_ in os.listdir(save_array_dir) if pattern.fullmatch(file_) ])
print(FILES)
c_constwind_avg = np.load(os.path.join(save_array_dir,'constwind_0.15.npy'))
c_constwind_avg = 0
for file_ in FILES:
    color = color=next(ax1._get_lines.prop_cycler)['color']
    c_avg = np.load(os.path.join(save_array_dir,file_))
    N_period = int(pattern.fullmatch(file_).group(1))
    omega = 2*np.pi/N_period


    ax1.plot(c_avg-c_constwind_avg, label = r'$\frac{\omega}{<a>} = 2\pi$/'+str(N_period), color = color)
    if N_period >100:
        x = range(Nt)
        y = [alpha_x0 + d_alpha_x*np.sin(omega*nt) for nt in range(Nt) ]
        ax2.plot(x,y, color = color, label = r'$\frac{\omega}{<a>} = 2\pi$/'+str(N_period))

pattern_constwind = re.compile( r'''constwind_(\d*\.*\d+)\.npy''')
FILES_constwind = tuple([ file_ for file_ in os.listdir(save_array_dir) if pattern_constwind.fullmatch(file_) ])
print(FILES_constwind)
for file_ in FILES_constwind:
    c_avg = np.load(os.path.join(save_array_dir,file_))
    alpha_x0 = float(pattern_constwind.fullmatch(file_).group(1))

    ax1.plot(c_avg - c_constwind_avg, label = r'Constant wind, $\alpha$ = {}'.format(alpha_x0), linestyle = '--')

ax1.legend()
ax2.legend()
plt.savefig(os.path.join(save_array_dir,'pop_plot.png'),bbox_inches="tight")
plt.show()

    