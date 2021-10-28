# Survival of the luckiest
#%%
import random, os
import numpy as np
from timeit import default_timer as timer
import imageio, cv2
import multiprocessing
from itertools import repeat

import custom_plots     # Custom library with plotting functions
from init import *      # Parsing all parameters and general infos from ini file 'parameters.ini' through 'init.py'

# TO DO:
# Check conservation laws
# Vectorise loops


def ADR(N_period, c0, a_chem):
    def grad_x_vector(M):
        return 0.5*(np.roll(M,1, axis=1) - np.roll(M,-1, axis=1))
    def lapl_2D_vector(M):
        sides = np.roll(M,1, axis=0) + np.roll(M,-1, axis=0) + np.roll(M,1, axis=1) + np.roll(M,-1, axis=1)
        corners = np.roll(M,[1,1],axis=(0,1)) + np.roll(M,[-1,1],axis=(0,1)) + \
            np.roll(M,[1,-1],axis=(0,1)) + np.roll(M,[-1,-1],axis=(0,1))
        return -3*M + 0.5*sides + 0.25*corners
    def AD_propagator_vector(M):
        return M + alpha_x*grad_x_vector(M) + delta*lapl_2D_vector(M)
    def Chem_propagator_vector(M):
        return M*np.exp(a_chem*dt) / ( 1 + M*(np.exp(a_chem*dt) - 1)*(b/a_chem) )
    # Advection & Diffusion
    period = dt*N_period    
    omega = 2*np.pi/period
    # Initialise matrices
    fft_section_0 = np.zeros((Nt,Nx))       # Section for k_y = 0 of the |FT| in k_x,t plane
    fft_section_1 = np.zeros((Nt,Nx))       # Sect. for n_k_y = dnk1

    fft_section_0_nowind = np.zeros((Nt,Nx))    # Same as above but for the nowind sys.
    fft_section_1_nowind = np.zeros((Nt,Nx))
    filenames= []
    c=c0.copy()     # Population subjected to sinusoidal wind
    c_nowind=c0.copy()

    c_avg = []      # Average population for each t
    c_nowind_avg = []

    # Time evolution: SWSS
    for nt in range(Nt):
        alpha_x = alpha_x0 + d_alpha_x*np.sin(omega*nt*dt)      # Sinusoidal Wind

        c_ad = AD_propagator_vector(c)
        c_chem = Chem_propagator_vector(c)
        c = 0.5 * (AD_propagator_vector(c_chem) + Chem_propagator_vector(c_ad))

        # Same as above but for CONSTANT WIND
        alpha_x = alpha_x0      

        c_ad = AD_propagator_vector(c_nowind)
        c_chem = Chem_propagator_vector(c_nowind)
        c_nowind = 0.5 * (AD_propagator_vector(c_chem) + Chem_propagator_vector(c_ad))
        
        
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
            kwargs = {'c':c,'c0':c0,'fft':fft,'c_nowind':c_nowind, 'fft0':fft0,\
                'alpha_x0':alpha_x0,'d_alpha_x':d_alpha_x, 'omega':omega,'dt':dt,'Nt':Nt,'nt':nt,\
                'filename' : f'{nt}.png', 'savefig_dir':savefig_dir}
            
            custom_plots.FirstPlot(**kwargs)

            filenames.append(kwargs['filename'])
        '''if nt % 100 == 0:    
            print(nt)'''


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


if __name__ == '__main__':

    timer_start = timer()

    print(f'alpha_0 = {alpha_x0}, d_alpha = {d_alpha_x}, omega = 2pi/{N_period}')

    random.seed(rand_seed)
    np.random.seed(rand_seed)

    # Create Savefig Directory if not present and delete previouses figures
    if not os.path.exists(savefig_dir):
        os.mkdir(savefig_dir)

    # Chemistry
    a_chem = np.random.uniform(low = a_chem_m-a_chem_range, high = a_chem_m+a_chem_range, size = (Ny,Nx))

    # Population
    c0 = np.random.uniform(low = c_m - c_range, high = c_m+c_range, size = (Ny,Nx))

    with multiprocessing.Pool(processes=len(N_period)) as pool:
        pool.starmap(ADR, zip(N_period,repeat(c0), repeat(a_chem)))

    print(f'Total Time elapsed: {timer()-timer_start:.1f}s')