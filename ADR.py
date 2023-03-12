# Survival of the luckiest
#%%
from math import ceil
import random, os
import numpy as np
from timeit import default_timer as timer
import imageio, cv2

import custom_plots     # Custom library with plotting functions
from init import ADR_params_dict      # Parsing all parameters and general infos from ini file 'parameters.ini' through 'init.py'

def grad_x_vector(M):
    return 0.5*(np.roll(M,1, axis=1) - np.roll(M,-1, axis=1))

def lapl_2D_vector(M):
    sides = np.roll(M,1, axis=0) + np.roll(M,-1, axis=0) + np.roll(M,1, axis=1) + np.roll(M,-1, axis=1)
    corners = np.roll(M,[1,1],axis=(0,1)) + np.roll(M,[-1,1],axis=(0,1)) + \
        np.roll(M,[1,-1],axis=(0,1)) + np.roll(M,[-1,-1],axis=(0,1))
    return -3*M + 0.5*sides + 0.25*corners

def AD_propagator_vector(M, alpha_x, delta):
    return M + alpha_x*grad_x_vector(M) + delta*lapl_2D_vector(M)

def Chem_propagator_vector(M, a_chem, b, dt):
    return M*np.exp(a_chem*dt) / ( 1 + M*(np.exp(a_chem*dt) - 1)*(b/a_chem) )

def ADR(*, Nx: int, Ny : int, Nt : int, N_period : int, 
        dt : float, alpha_x0 : float, d_alpha_x : float, 
        delta : float, b : float, a_chem_m : float, a_chem_range : float, 
        c_m : float, c_range: float, 
        savefig_dir : str, gifname : str, fname_c_avg :str, fname_c_constwind_avg:str,
        dnk1 : int, 
        save_c_avg : bool, save_c_constwind_avg : bool, show_3Dplot : bool, save_array_dir : str,
        second_plot_name : str, rand_seed : int, make_first_plot:bool, 
        N_t_stationary_min : int, stationary_start : bool, 
        stat_pops_dir : str, stationary_c_fname : str, stationary_c_constwind_fname : str):

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

    # Chemistry
    a_chem = np.random.uniform(low = a_chem_m-a_chem_range, high = a_chem_m+a_chem_range, size = (Ny,Nx))

    # Stationary condition is considered reached for the first nt > N_t_stationary_min and multiple of N_period (so that the wind phase is 0)
    N_t_stationary = N_period * ceil(N_t_stationary_min/N_period)

    # Population
    # Stationary start is chosen by the user, but if files are not present, stationary_initial_cond is still False
    # It would be nice to decouple sinusoidal and constant wind files. Here we need both
    stationary_initial_cond = os.path.exists(stationary_c_fname) and os.path.exists(stationary_c_constwind_fname) and stationary_start
    if stationary_initial_cond:    # If stationary files are present and user chose stationary intial cond.
        c = np.load(stationary_c_fname)
        c_constwind = np.load(stationary_c_constwind_fname)
    else:
        c = np.random.uniform(low = c_m - c_range, high = c_m+c_range, size = (Ny,Nx))
        c_constwind=c.copy()
        Nt += N_t_stationary # If files are not present, we create it first and than proceed normally when stationariety is reached

    # Initialise matrices
    fft_section_0 = np.zeros((Nt,Nx))       # Section for k_y = 0 of the |FT| in k_x,t plane
    fft_section_1 = np.zeros((Nt,Nx))       # Sect. for n_k_y = dnk1

    fft_section_0_constwind = np.zeros((Nt,Nx))    # Same as above but for the constwind sys.
    fft_section_1_constwind = np.zeros((Nt,Nx))
    filenames= []


    c_avg = []      # Average population for each t
    c_constwind_avg = []

    # Time evolution: SWSS
    for nt in range(Nt):
        alpha_x = alpha_x0 + d_alpha_x*np.sin(omega*nt*dt)      # Sinusoidal Wind


        c_ad = AD_propagator_vector(c, alpha_x, delta)
        c_chem = Chem_propagator_vector(c, a_chem, b, dt)
        c = 0.5 * (AD_propagator_vector(c_chem, alpha_x, delta) + Chem_propagator_vector(c_ad, a_chem, b, dt))

        # Same as above but for CONSTANT WIND
        alpha_x = alpha_x0      

        c_ad = AD_propagator_vector(c_constwind, alpha_x, delta)
        c_chem = Chem_propagator_vector(c_constwind, a_chem, b, dt)
        c_constwind = 0.5 * (AD_propagator_vector(c_chem, alpha_x, delta) + Chem_propagator_vector(c_ad, a_chem, b, dt))
        
        # Saving stationary arrays if we didn't start from stationariety
        if nt == N_t_stationary and not stationary_initial_cond: 
            np.save(stationary_c_fname, c) # Note that one of them can be already existing ad is overwritten 
            np.save(stationary_c_constwind_fname, c_constwind)
        
        # Update average population
        if nt >= N_t_stationary or not stationary_start: # If user didn't request stationary start we save also the transient population
            c_avg.append(c.mean())
            c_constwind_avg.append(c_constwind.mean())            
        
        
        # Fourier transform of population
        fft=np.fft.fft2(c)      
        # Placing origin at the center...
        x1,x2, x3, x4= fft[:Ny//2,:Nx//2], fft[Ny//2:,:Nx//2], fft[:Ny//2,Nx//2:], fft[Ny//2:,Nx//2:]
        fft1=np.concatenate((x3, x1), axis=1)
        fft2=np.concatenate((x4, x2), axis=1)
        fft=np.concatenate((fft2, fft1), axis=0)

        fft0=np.fft.fft2(c_constwind)
        x1,x2, x3, x4= fft0[:Ny//2,:Nx//2], fft0[Ny//2:,:Nx//2], fft0[:Ny//2,Nx//2:], fft0[Ny//2:,Nx//2:]
        fft1=np.concatenate((x3, x1), axis=1)
        fft2=np.concatenate((x4, x2), axis=1)
        fft0=np.concatenate((fft2, fft1), axis=0)

        # Averaging along k_y in order to reduce noise
        fft_section_0[nt][:] = np.mean(np.abs(fft[Ny//2 - 1:Ny//2 +1][:]), axis = 0)
        fft_section_1[nt][:] = np.mean(np.abs(fft[Ny//2 + dnk1 - 1:Ny//2 + dnk1 +1][:]), axis = 0)

        fft_section_0_constwind[nt][:] = np.mean(np.abs(fft0[Ny//2 - 1:Ny//2 +1][:]), axis = 0)
        fft_section_1_constwind[nt][:] = np.mean(np.abs(fft0[Ny//2 + dnk1 - 1:Ny//2 + dnk1 +1][:]), axis = 0)
        #fft_section_0[nt][:] = scipy.signal.savgol_filter(tmp, 5, 3)
        
        # PLOT of c, |FT[c]|, c_0, |FT[c_0]|, (c-c_0)/c_0, |FT[c]|/|FT[c_0]|
        if make_first_plot and nt % 5 == 0:
            kwargs = {'c':c,'vmax':c_m+c_range, 'vmin':c_m-c_range ,'fft':fft,'c_constwind':c_constwind, 'fft0':fft0,\
                'alpha_x0':alpha_x0,'d_alpha_x':d_alpha_x, 'omega':omega,'dt':dt,'Nt':Nt,'nt':nt,\
                'filename' : f'{nt}.png', 'savefig_dir':savefig_dir}
            
            custom_plots.FirstPlot(**kwargs)

            filenames.append(kwargs['filename'])
        if nt % 100 == 0:    
            print(nt)


    # Save arrays of average population, to be analysed later by pop_plotter.py
    if save_c_avg or save_c_constwind_avg:
        if not os.path.exists(save_array_dir):
            os.mkdir(save_array_dir)
        if save_c_avg:
            np.save( os.path.join(save_array_dir,fname_c_avg.format(N_period)) , c_avg)
        if save_c_constwind_avg:
            np.save( os.path.join(save_array_dir,fname_c_constwind_avg.format(alpha_x0)) , c_constwind_avg)

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

    Z_0_constwind = np.log10(fft_section_0_constwind[:][:].transpose())
    Z_1_constwind = np.log10(fft_section_1_constwind[:][:].transpose())
    Z_0_blur_constwind = cv2.GaussianBlur(Z_0_constwind, (7, 7), 0)
    Z_1_blur_constwind = cv2.GaussianBlur(Z_1_constwind, (7, 7), 0)

    # 3D PLOT of fft_section_0
    if show_3Dplot:
        custom_plots.ThreeD_Plot(Nt=Nt, Nx=Nx, Z_0=Z_0)

    #%%
    # PLOT on k_x,t plane of |FT[c]|, |FT[c_0]|, |FT[c]|/|FT[c_0]|
    kwargs2 = {'Nt':Nt, 'N_period':N_period, 'Z_0_blur':Z_0_blur, \
        'Z_1_blur':Z_1_blur, 'alpha_x0':alpha_x0, 'd_alpha_x':d_alpha_x,\
        'omega':omega, 'dt':dt, 'Z_0_blur_constwind':Z_0_blur_constwind, \
        'Z_1_blur_constwind':Z_1_blur_constwind, 'second_plot_name':second_plot_name, 'savefig_dir' : savefig_dir}
    custom_plots.SecondPlot(**kwargs2)

    # Print total time elapsed
    print(f'Time elapsed: {timer()-timer_start:.1f}s')

if __name__ == '__main__':

    ADR(**ADR_params_dict)