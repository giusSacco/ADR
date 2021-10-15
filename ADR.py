# Survival of the luckiest
#%%
import random, os
import numpy as np
from timeit import default_timer as timer
import imageio, cv2

import custom_plots     # Custom library with plotting functions
from init import *      # Parsing all parameters and general infos from ini file 'parameters.ini' through 'init.py'

# To do:
# Nowind should not be computed if not necessary

timer_start = timer()

print(f'alpha_0 = {alpha_x0}, d_alpha = {d_alpha_x}, omega = 2pi/{N_period}')

random.seed(rand_seed)
np.random.seed(rand_seed)

# Create Savefig Directory if not present
if not os.path.exists(savefig_dir):
    os.mkdir(savefig_dir)
# Create directory where stationary populations are saved
if not os.path.exists(stat_pops_dir):
    os.mkdir(stat_pops_dir)

# Advection & Diffusion
period = dt*N_period    
omega = 2*np.pi/period

def grad_x(M,i,j):
    return 0.5*( M[i][(j+1)%Nx] - M[i][(j-1)%Nx] ) 

def lapl_2D(M,i,j):
    sides = M[(i+1)%Ny][j] + M[(i-1)%Ny][j] + M[i][(j+1)%Nx] + M[i][(j-1)%Nx]
    corners = M[(i+1)%Ny][(j+1)%Nx] + M[(i-1)%Ny][(j+1)%Nx] + M[(i+1)%Ny][(j-1)%Nx] + M[(i-1)%Ny][(j-1)%Nx]
    return -3*M[i][j] + 0.5*sides + 0.25*corners

def AD_propagator(M,i,j):
    return M[i][j] + alpha_x*grad_x(M,i,j) + delta*lapl_2D(M,i,j)

# Chemistry
a_chem = np.random.uniform(low = a_chem_m-a_chem_range, high = a_chem_m+a_chem_range, size = (Ny,Nx))

def Chem_propagator(M,i,j):
    return M[i][j]*np.exp(a_chem[i][j]*dt) / ( 1 + M[i][j]*(np.exp(a_chem[i][j]*dt) - 1)*(b/a_chem[i][j]) )

# Population
if os.path.exists(stationary_c_fname) and os.path.exists(stationary_c_constwind_fname) and stationary_start:    # If stationary files are present
    c = np.loadtxt(stationary_c_fname)
    c_nowind = np.loadtxt(stationary_c_constwind_fname)
else:
    c = np.random.uniform(low = c_m - c_range, high = c_m+c_range, size = (Ny,Nx))
    c_nowind=c.copy()

# Initialise matrices
fft_section_0 = np.zeros((Nt,Nx))       # Section for k_y = 0 of the |FT| in k_x,t plane
fft_section_1 = np.zeros((Nt,Nx))       # Sect. for n_k_y = dnk1

fft_section_0_nowind = np.zeros((Nt,Nx))    # Same as above but for the nowind sys.
fft_section_1_nowind = np.zeros((Nt,Nx))
filenames= []

c_chem = np.zeros((Ny,Nx))  # Used for SWSS
c_ad = np.zeros((Ny,Nx))

c_avg = []      # Average population for each t
c_nowind_avg = []


# Time evolution: SWSS
for nt in range(Nt):
    alpha_x = alpha_x0 + d_alpha_x*np.sin(omega*nt*dt)      # Sinusoidal Wind
    for i in range(Ny):     # Vectorize maybe?
        for j in range(Nx):
            c_chem[i][j] = Chem_propagator(c,i,j)
            c_ad[i][j] = AD_propagator(c,i,j)
    for i in range(Ny):
        for j in range(Nx):
            c[i][j] = ( AD_propagator(c_chem,i,j) + Chem_propagator(c_ad,i,j) ) / 2

    alpha_x = alpha_x0      # Same as above but for constant wind
    for i in range(Ny):
        for j in range(Nx):
            c_chem[i][j] = Chem_propagator(c_nowind,i,j)
            c_ad[i][j] = AD_propagator(c_nowind,i,j)
    for i in range(Ny):
        for j in range(Nx):
            c_nowind[i][j] = ( AD_propagator(c_chem,i,j) + Chem_propagator(c_ad,i,j) ) / 2

    # Saving stationary arrays if they don't already exist
    if nt == N_t_stationary: 
        if not os.path.exists(stationary_c_fname):
            np.savetxt(stationary_c_fname, c)
        if not os.path.exists(stationary_c_constwind_fname):
            np.savetxt(stationary_c_constwind_fname, c_nowind)
    
    # Update average population
    c_avg.append(c.mean())
    c_nowind_avg.append(c_nowind.mean())            
    
    
    # Fourier transform of population
    fft=np.fft.fft2(c)      
    # Placing origin at the center...
    x1,x2, x3, x4= fft[:Ny//2,:Nx//2], fft[Ny//2:,:Nx//2], fft[:Ny//2,Nx//2:], fft[Ny//2:,Nx//2:]
    fft1=np.concatenate((x3, x1), axis=1)
    fft2=np.concatenate((x4, x2), axis=1)
    fft=np.concatenate((fft2, fft1), axis=0)

    fft0=np.fft.fft2(c_nowind)
    x1,x2, x3, x4= fft0[:Ny//2,:Nx//2], fft0[Ny//2:,:Nx//2], fft0[:Ny//2,Nx//2:], fft0[Ny//2:,Nx//2:]
    fft1=np.concatenate((x3, x1), axis=1)
    fft2=np.concatenate((x4, x2), axis=1)
    fft0=np.concatenate((fft2, fft1), axis=0)

    # Averaging along k_y in order to reduce noise
    fft_section_0[nt][:] = np.mean(np.abs(fft[Ny//2 - 1:Ny//2 +1][:]), axis = 0)
    fft_section_1[nt][:] = np.mean(np.abs(fft[Ny//2 + dnk1 - 1:Ny//2 + dnk1 +1][:]), axis = 0)

    fft_section_0_nowind[nt][:] = np.mean(np.abs(fft0[Ny//2 - 1:Ny//2 +1][:]), axis = 0)
    fft_section_1_nowind[nt][:] = np.mean(np.abs(fft0[Ny//2 + dnk1 - 1:Ny//2 + dnk1 +1][:]), axis = 0)
    #fft_section_0[nt][:] = scipy.signal.savgol_filter(tmp, 5, 3)
    
    # PLOT of c, |FT[c]|, c_0, |FT[c_0]|, (c-c_0)/c_0, |FT[c]|/|FT[c_0]|
    if make_first_plot and nt % 5 == 0:
        kwargs = {'c':c,'vmax':c_m+c_range, 'vmin':c_m-c_range ,'fft':fft,'c_nowind':c_nowind, 'fft0':fft0,\
            'alpha_x0':alpha_x0,'d_alpha_x':d_alpha_x, 'omega':omega,'dt':dt,'Nt':Nt,'nt':nt,\
            'filename' : f'{nt}.png', 'savefig_dir':savefig_dir}
        
        custom_plots.FirstPlot(**kwargs)

        filenames.append(kwargs['filename'])
    if nt % 5 == 0:    
        print(nt)


# Save arrays of average population, to be analysed later by pop_plotter.py
if save_c_avg or save_c_nowind_avg:
    if not os.path.exists(save_array_dir):
        os.mkdir(save_array_dir)
    if save_c_avg:
        np.savetxt( os.path.join(save_array_dir,fname_c_avg.format(N_period)) , c_avg)
    if save_c_nowind_avg:
        np.savetxt( os.path.join(save_array_dir,fname_c_nowind_avg.format(alpha_x0)) , c_nowind_avg)

# Create gif with previouses images
if make_first_plot:
    with imageio.get_writer(os.path.join(savefig_dir,gifname), mode='I', fps = 2) as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(savefig_dir,filename))
            writer.append_data(image)

#%%   
# Gaussain blur and log scale for fft
dk1 = (2*np.pi/Ny)*dnk1     # Circa pi/4

Z_0 = np.log10(fft_section_0[:][:].transpose())
Z_1 = np.log10(fft_section_1[:][:].transpose())
Z_0_blur = cv2.GaussianBlur(Z_0, (7, 7), 0)
Z_1_blur = cv2.GaussianBlur(Z_1, (7, 7), 0)

Z_0_nowind = np.log10(fft_section_0_nowind[:][:].transpose())
Z_1_nowind = np.log10(fft_section_1_nowind[:][:].transpose())
Z_0_blur_nowind = cv2.GaussianBlur(Z_0_nowind, (7, 7), 0)
Z_1_blur_nowind = cv2.GaussianBlur(Z_1_nowind, (7, 7), 0)

# 3D PLOT of fft_section_0
if show_3Dplot:
    custom_plots.ThreeD_Plot(Nt=Nt, Nx=Nx, Z_0=Z_0)

#%%
# PLOT on k_x,t plane of |FT[c]|, |FT[c_0]|, |FT[c]|/|FT[c_0]|
kwargs2 = {'Nt':Nt, 'N_period':N_period, 'Z_0_blur':Z_0_blur, \
    'Z_1_blur':Z_1_blur, 'alpha_x0':alpha_x0, 'd_alpha_x':d_alpha_x,\
    'omega':omega, 'dt':dt, 'Z_0_blur_nowind':Z_0_blur_nowind, \
    'Z_1_blur_nowind':Z_1_blur_nowind, 'second_plot_name':second_plot_name, 'savefig_dir' : savefig_dir}
custom_plots.SecondPlot(**kwargs2)


# Print total time elapsed
print(f'Time elapsed: {timer()-timer_start:.1f}s')