
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2, os
import matplotlib.gridspec as gridspec

# PLOT of c, |FT[c]|, c_0, |FT[c_0]|, (c-c_0)/c_0, |FT[c]|/|FT[c_0]|
def FirstPlot(*,c,c0,fft,c_nowind, fft0, alpha_x0,d_alpha_x, omega, dt, Nt, nt, savefig_dir, filename):
    plt.figure(figsize=(6,4))
    plt.subplots_adjust(left=None, bottom=None, right=1.5, top=1.1, wspace=None, hspace=None)
    gs = gridspec.GridSpec(3, 3, height_ratios=[3, 3,1])

    # plot c
    ax1 = plt.subplot(gs[0,0])
    ax1.set_title('Sinusoidal wind')
    ax1.set_ylabel('$N_y$')
    ax1.set_xlabel('$N_x$')
    plot1=ax1.imshow(c, vmax=c0.max(), vmin=c0.min())
    plt.colorbar(plot1, ax=ax1, label='c')

    # plot |FT[c]|
    ax2 = plt.subplot(gs[1,0])
    #plot2=ax2.imshow(np.abs(fft), extent=[-np.pi, np.pi, -np.pi, np.pi], vmax = 10)
    ax2.set_xlabel('$K_x$')
    ax2.set_ylabel('$K_y$')
    tmp_plot2=np.log10(np.abs(fft))
    plot2=ax2.imshow(cv2.GaussianBlur(tmp_plot2,(3,3),0), extent=[-np.pi, np.pi, -np.pi, np.pi],vmax= 3, vmin=0)
    plt.colorbar(plot2, ax=ax2, label='log(|FT[c]|)')
    
    #plot c_0
    ax3 = plt.subplot(gs[0,1])
    plot1=ax3.imshow(c_nowind, vmax=c0.max(), vmin=c0.min())
    ax3.set_title('Constant wind')
    ax3.set_xlabel('$N_x$')
    ax3.set_ylabel('$N_y$')
    plt.colorbar(plot1, ax=ax3,label='$c_0$')
    
    # plot |FT[c_0]|
    ax4 = plt.subplot(gs[1,1])
    ax4.set_xlabel('$K_x$')
    ax4.set_ylabel('$K_y$')
    #plot2=ax2.imshow(np.abs(fft), extent=[-np.pi, np.pi, -np.pi, np.pi], vmax = 10)
    tmp_plot4=np.log10(np.abs(fft0))
    plot4=ax4.imshow(cv2.GaussianBlur(tmp_plot4,(3,3),0), extent=[-np.pi, np.pi, -np.pi, np.pi],vmax= 3, vmin=0)
    plt.colorbar(plot4, ax=ax4,label='$log(|FT[c_0]|)$')

    # plot alpha(t)
    ax5 = plt.subplot(gs[2,0])
    x = range(Nt)
    y = [alpha_x0 + d_alpha_x*np.sin(omega*nt*dt) for nt in range(Nt) ]
    plt.plot(x,y)
    ax5.set_ylabel(r'$\alpha$')
    ax5.set_xlabel('$n_t$')
    ax5.set_xlim(0,Nt)
    ax5.set_ylim(alpha_x0-d_alpha_x,alpha_x0+d_alpha_x)
    ax5.scatter(nt,y[nt],  s=50, color = 'red')
    #plot alpha_const
    ax6 = plt.subplot(gs[2,1])
    x = range(Nt)
    y = [alpha_x0 + d_alpha_x*np.sin(omega*nt*dt) for nt in range(Nt) ]
    plt.plot(x,np.ones(Nt)*alpha_x0)
    ax6.set_ylabel(r'$\alpha$')
    ax6.set_xlabel('$n_t$')
    ax6.set_xlim(0,Nt)
    ax6.set_ylim(alpha_x0-d_alpha_x,alpha_x0+d_alpha_x)
    ax6.scatter(nt,alpha_x0,  s=50, color = 'red')
    
    #plot (c-c_0)/c_0
    ax1 = plt.subplot(gs[0,2])
    ax1.set_title('Sinusoidal/constant')
    ax1.set_xlabel('$N_x$')
    ax1.set_ylabel('$N_y$')
    plot1=ax1.imshow( (c-c_nowind)/c_nowind ,vmax=.4, vmin=-.4,cmap = 'RdBu')
    plt.colorbar(plot1, ax=ax1,label=r'$\frac{c-c_0}{c_0}$')

    # plot |FT[c]|/|FT[c_0]|
    ax2 = plt.subplot(gs[1,2])
    #plot2=ax2.imshow(np.abs(fft), extent=[-np.pi, np.pi, -np.pi, np.pi], vmax = 10)
    ax2.set_xlabel('$K_x$')
    ax2.set_ylabel('$K_y$')
    tmp_plot6=tmp_plot2-tmp_plot4
    plot2=ax2.imshow(cv2.GaussianBlur(tmp_plot6,(3,3),0), extent=[-np.pi, np.pi, -np.pi, np.pi], vmax=1,vmin=-1,cmap = 'RdBu')
    plt.colorbar(plot2, ax=ax2,label='$log(|FT[c]|/|FT[c_0]|)$')

    #plot alpha/alpha_0
    ax5 = plt.subplot(gs[2,2])
    x = range(Nt)
    y = [(alpha_x0 + d_alpha_x*np.sin(omega*nt*dt))/alpha_x0 for nt in range(Nt) ]
    plt.plot(x,y)
    ax5.set_xlabel('$n_t$')
    ax5.set_ylabel(r'$\alpha/\alpha_0$')
    ax5.set_xlim(0,Nt)
    ax5.set_ylim((alpha_x0-d_alpha_x)/alpha_x0,(alpha_x0+d_alpha_x)/alpha_x0)
    ax5.scatter(nt,y[nt],  s=50, color = 'red')
    
    plt.savefig(os.path.join(savefig_dir,filename),bbox_inches="tight")
    plt.close()


def ThreeD_Plot(Nt,Nx,Z_0):
    x = range(0, Nt)
    y = range(0, Nx)
    X, Y = np.meshgrid(x, y)
    # Normalize to [0,1]
    norm = plt.Normalize(Z_0.min(), Z_0.max())
    colors = cm.viridis(norm(Z_0))
    rcount, ccount, _ = colors.shape

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('$n_t$')
    ax.set_ylabel('x')
    ax.set_zlabel('fft')
    #ax.set_zlim([0,20])
    surf = ax.plot_surface(X, Y, Z_0, rcount=1, ccount=50, cmap=cm.coolwarm, shade=False)
    surf.set_facecolor((0,0,0,0))

    plt.show()


def SecondPlot(Nt, N_period, Z_0_blur, Z_1_blur, alpha_x0, d_alpha_x, omega, dt, Z_0_blur_nowind, Z_1_blur_nowind, second_plot_name, savefig_dir):
    
    Z_max = max(Z_0_blur.max(), Z_1_blur.max()) # Servono per i limiti della colorbar
    Z_min = min(Z_0_blur.min(), Z_1_blur.min())
    
    plt.figure(figsize=(4,4*1.2))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2,3.2, 1])

    ax1 = plt.subplot(gs[0,0])
    im = ax1.imshow(Z_0_blur, aspect='auto', vmin = Z_min, vmax=Z_max, extent=[ 0, Nt, -np.pi, np.pi])
    periods = range(0,Nt,N_period)[1:]
    ax1.vlines(periods,ymin = -np.pi,ymax=np.pi, colors='k', linestyles='dashed')
    ax1.set_ylabel(r'$K_x$')
    ax1.set_title(r'Sinusiodal Wind, $K_y$ = 0')
    #ax1.set_xlim(0,Nt-1)
    #ax1.set_ylim(0,Nx)
    ax1.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')
    #plt.colorbar(im)

    ax3 = plt.subplot(gs[1,0])
    im = ax3.imshow(Z_1_blur, aspect='auto',vmin = Z_min, vmax=Z_max,extent=[ 0, Nt, -np.pi, np.pi])
    periods = range(0,Nt,N_period)[1:]
    ax3.set_ylabel(r'$K_x$')
    ax3.vlines(periods,ymin = -np.pi, ymax=np.pi, colors='k', linestyles='dashed')
    ax3.set_title(f'$K_y = \pi/4$')
    ax3.set_xlim(0,Nt-1)
    ax3.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')
    plt.colorbar(im, location = 'bottom', label = 'log(FT[c])')

    ax2 = plt.subplot(gs[2,0])
    x = range(Nt)
    y = [alpha_x0 + d_alpha_x*np.sin(omega*nt*dt) for nt in range(Nt) ]
    plt.plot(x,y)
    ax2.vlines(periods,ymin = alpha_x0-d_alpha_x,ymax=alpha_x0+d_alpha_x, colors='k', linestyles='dashed')
    ax2.set_ylabel(r'$\alpha$')
    ax2.set_xlabel('$n_t$')
    ax2.set_xlim(0,Nt-1)
    ax2.set_ylim(alpha_x0-d_alpha_x,alpha_x0+d_alpha_x)

    # NOWIND PLOT

    ax1 = plt.subplot(gs[0,1])
    im = ax1.imshow(Z_0_blur_nowind, aspect='auto', vmin = Z_min, vmax=Z_max,extent=[ 0, Nt, -np.pi, np.pi])
    periods = range(0,Nt,N_period)[1:]
    ax1.vlines(periods,ymin = -np.pi, ymax=np.pi, colors='k', linestyles='dashed')
    ax1.set_title('Constant wind, $K_y$ = 0')
    ax1.set_xlim(0,Nt-1)

    ax1.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')

    ax3 = plt.subplot(gs[1,1])
    im = ax3.imshow(Z_1_blur_nowind, aspect='auto',vmin = Z_min, vmax=Z_max,extent=[ 0, Nt, -np.pi, np.pi])
    periods = range(0,Nt,N_period)[1:]
    ax3.vlines(periods,ymin = -np.pi, ymax=np.pi, colors='k', linestyles='dashed')
    ax3.set_title(f'$K_y = \pi/4$')
    ax3.set_xlim(0,Nt-1)

    ax3.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')
    plt.colorbar(im, location = 'bottom', label = '$log(FT[c_0])$')

    ax2 = plt.subplot(gs[2,1])
    x = range(Nt)
    y = [alpha_x0 + d_alpha_x*np.sin(omega*nt*dt) for nt in range(Nt) ]
    plt.plot(x,alpha_x0*np.ones(Nt))
    ax2.vlines(periods,ymin = alpha_x0-d_alpha_x,ymax=alpha_x0+d_alpha_x, colors='k', linestyles='dashed')
    ax2.set_xlabel('$n_t$')
    ax2.set_xlim(0,Nt-1)
    ax2.set_ylim(alpha_x0-d_alpha_x,alpha_x0+d_alpha_x)

    # PLOT DIFFERENCE

    ax1 = plt.subplot(gs[0,2])
    im = ax1.imshow(Z_0_blur - Z_0_blur_nowind, aspect='auto', vmin = -.5, vmax=.5,cmap = 'RdBu',extent=[ 0, Nt, -np.pi, np.pi])
    periods = range(0,Nt,N_period)[1:]
    ax1.vlines(periods,ymin = -np.pi, ymax=np.pi, colors='k', linestyles='dashed')
    ax1.set_title('Sinusoidal/constant, $K_y$ = 0')
    ax1.set_ylabel(r'$K_x$')
    ax1.set_xlim(0,Nt-1)

    ax1.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')

    ax3 = plt.subplot(gs[1,2])
    im = ax3.imshow(Z_1_blur - Z_1_blur_nowind, aspect='auto',cmap = 'RdBu',vmin=-0.5,vmax=0.5,extent=[ 0, Nt, -np.pi, np.pi])
    periods = range(0,Nt,N_period)[1:]
    ax3.vlines(periods,ymin = -np.pi, ymax=np.pi, colors='k', linestyles='dashed')
    ax3.set_title(f'$K_y = \pi/4$')
    ax3.set_ylabel(r'$K_x$')
    ax3.set_xlim(0,Nt-1)

    ax3.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')
    plt.colorbar(im, location = 'bottom', label = '$log(|FT(c)|/|FT(c_0)|)$')

    ax2 = plt.subplot(gs[2,2])
    x = range(Nt)
    y = [alpha_x0 + d_alpha_x*np.sin(omega*nt*dt) for nt in range(Nt) ]
    plt.plot(x,y)
    ax2.vlines(periods,ymin = alpha_x0-d_alpha_x,ymax=alpha_x0+d_alpha_x, colors='k', linestyles='dashed')
    ax2.set_xlabel('$n_t$')
    ax2.set_ylabel(r'$\alpha$')
    ax2.set_xlim(0,Nt)
    ax2.set_ylim(alpha_x0-d_alpha_x,alpha_x0+d_alpha_x)
    #plt.tight_layout()
    plt.savefig(os.path.join(savefig_dir,second_plot_name),bbox_inches="tight")
    plt.close()