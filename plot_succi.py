#! /usr/bin/env python3

import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import sys, os
import re
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from cmath import rect

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

dt = 0.01
name_pattern = re.compile(r'omega_([0-9.]+).txt')

def read(fname, Nper):
    ydata = np.loadtxt(fname)

    ntot, residual = divmod(len(ydata), Nper)

    if ntot > 1:
        ydata = ydata[int((ntot-1)*Nper):int(ntot*Nper)]
    else:
        ydata = ydata[int(ntot*Nper):]
    #print(f'{Nper=}, {ntot=}, {residual=}')

    assert abs(len(ydata) - int(Nper)) < 2, f'{Nper=}, {len(ydata)=}'
    
    x=np.arange(len(ydata))*dt
    #print(len(ydata))
    return x, ydata

def analyze(omega, y):

    mean = np.mean(y)
    y_ref = y-mean
    A = 0.5*(np.max(y_ref) - np.min(y_ref))
    delta_phi = find_phase( y)

    return omega, mean, A, delta_phi
    
# def find_phase_OLD(omega, y):
#     #get amplitude from fft

#     #dt = (2*np.pi/omega) / 100 #Nper / 100

#     first_zero = np.argmin(np.abs(y)[:int(len(y)/2)])   *   dt
#     second_zero = np.argmin(np.abs(y)[int(len(y)/2+1):])*   dt

#     delta_phi = 0.5*(omega*first_zero+(omega*second_zero-np.pi))

#     return delta_phi * 180/np.pi #in teoria sarebbe - delta phi

def find_phase(y):
    signal = fft(y-np.mean(y))
    return np.angle(max(list(signal), key = np.abs)) * 180/np.pi

def do_fft(y, dt):

    signal = fft(y-np.mean(y))
    freq = fftfreq(len(y), d=dt)

    return freq , signal

def debye_real(x, tau, B, alpha):
    return - B/(1+(x*tau)**alpha)

def debye_imag(x, tau, B, alpha):
    return x*tau*B/(1+(x*tau)**alpha)

def hkr(x, tau, B, alpha, beta):
    return np.real(B/(1+(1j*x*tau)**alpha)**beta)

def hki(x, tau, B, alpha, beta):
    return np.imag(B/(1+(1j*x*tau)**alpha)**beta)

def hk_spectra(t, tau, B, alpha, beta):
    return 1/np.pi*((t/tau)**(alpha*beta)*np.sin(beta*theta(t, tau, B, alpha, beta))/((t/tau)**(alpha*2) + 2*np.cos(np.pi*alpha)*(t/tau)**(alpha)+1)**(0.5*beta))

def theta(t, tau, B, alpha, beta):
    return np.arctan2(np.sin(np.pi*alpha), np.cos(np.pi*alpha)*(t/tau)**(alpha))
    
def main():
    res_dir = 'NewArrays2022/'
    Res=[]
    Analysis = []
    for fname in glob(os.path.join(res_dir,'*.txt')):

        Nper= float(name_pattern.match(os.path.basename(fname)).groups()[0])

        if Nper < 5:
            continue
        res = read(fname, Nper)
        
        omega = 2*np.pi/(Nper*dt)

        Res.append(res)
        Analysis.append(analyze(omega, res[1]))

    plt.figure()
    for x, y in Res:
        plt.plot(x, y)

    # fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)

    fig2, (ax2, ax3) = plt.subplots(2, 1)

    for o, m, a, p in Analysis:

        # ax1.plot(o, m, 'o')
        # ax1.set_xscale('log')
        # ax1.set_ylabel('Mean Value')

        ax2.plot(o, a*100, 'ok')
        ax2.set_xscale('log')
        ax2.set_ylabel(r'Modulation Amplitude [\%]')

        ax3.plot(o, 180-(90-p), 'ok')
        ax3.set_xscale('log')
        ax3.set_ylabel('Relative Phase[°]')
        ax3.set_ylim([0, 180])

        ax3.set_xlabel(r'$\frac{\omega  a}{2 \pi}$')
    
    fig2_, (ax2_, ax3_) = plt.subplots(2, 1)

    real, imag, omg = [], [], []

    for o, m, a, p in Analysis:
        omg.append(o)
        p = 180-(90-p)

        com = rect(a, p*np.pi/180)
        # ax1.plot(o, m, 'o')
        # ax1.set_xscale('log')
        # ax1.set_ylabel('Mean Value')

        real.append(np.real(com))
        imag.append(np.imag(com))

        ax2_.plot(o, np.real(com), 'ok')
        ax2_.set_xscale('log')
        ax2_.set_ylabel(r'Real Part')

        ax3_.plot(o, np.imag(com), 'ok')
        ax3_.set_xscale('log')
        ax3_.set_ylabel('Imaginary Part')

        ax3_.set_xlabel(r'$\frac{\omega  a}{2 \pi}$')
    
    omg = np.array(omg)
    
    Param, _ = curve_fit(hki, omg, imag)
    Param2, _ = curve_fit(hkr, omg, real)

    print(Param)
    print(Param2)

    omg.sort()
    ax3_.plot(omg, hki(omg, *Param), '--r')
    ax2_.plot(omg, hkr(omg, *Param2), '--r')

    Pt = 0.5*(Param+Param2)

    fig2__, ax__ = plt.subplots(1, 1)
    xxx=np.logspace(-2, 2, 500)
    ax__.plot(xxx, np.abs(hk_spectra(xxx, *Pt)), 'k')
    ax__.set_xscale('log')
    ax__.set_ylabel(r'$g(\tau_{rel})$')
    ax__.set_xlabel(r'$\tau_{rel}/a$')



    #the fft was a failed experiment
    m=0
    spectra = []
    maxes=[]
    peaks=[]
    for _, y in Res:
        freq , signal = do_fft(y, dt)
        maxes.append(max(freq))
        peaks.append(abs(freq[np.argmax( np.abs(signal))]))
        if max(freq) > m:
            m=max(freq)
            # plt.figure()
            # plt.plot(freq[freq>=0] , np.abs(signal)[freq>=0])
            # plt.plot(np.linspace(0, m, num=int(m*1000)), spline(freq[freq>=0], np.abs(signal[freq>=0]), k=1)(np.linspace(0, m, num=int(m*1000))))
            # plt.show()
        try:
            spectra.append(spline(freq[freq>=0], np.abs(signal[freq>=0]), ext=1, k=1))
        except Exception:
            spectra.append(spline(np.linspace(0, m, num=10), np.zeros(10)))
    xf = np.linspace(0, m, num=int(m*5000))
    Img = []
    omegas = []
    for signal_spline, (omega, _,_,_) in zip(spectra, Analysis):
        omegas.append(omega/(2*np.pi))
        sig = signal_spline(xf)
        if np.max(sig)>0:
            Img.append(sig/np.max(sig))
        else:
            Img.append(np.zeros(len(xf)))
    
    omegas, Img, maxes, peaks, spectra, Res = zip(*sorted(zip(omegas, Img, maxes, peaks, spectra, Res), reverse=False))

    Img=np.array(Img)
    # 
    # fig3, ax = plt.subplots(1, 1)
    # ax.imshow(Img, extent=[0, m, max(omegas), min(omegas)], aspect='auto')
    # ax.imshow(Img, aspect='auto')
    # ax.set_yticks=omegas
    # ax.set_xticks=xf
    # #ax.set_yscale('log')
    # # ax.plot(maxes, omegas, 'or')
    # ax.plot(peaks, omegas, 'or')
    
    
    fig4, axn = plt.subplots(1, 1)
    axn.plot(omegas, peaks, 'or')
    axn.plot(omegas, omegas, '--k')
    axn.set_xlabel('Omega')
    axn.set_ylabel('FFT peak')


    index_to_plot_alone = -5 # 0 ,  len(omegas)//2 ,  -15
    fig5, ax_ = plt.subplots(1, 1)
    y = Res[index_to_plot_alone][1]
    x = np.linspace(0, 1, len(y))
    Params, _ = curve_fit(sine_fit, x, y, p0=[2*np.pi, 0.5*(max(y)-min(y)), 0, np.mean(y)])
    s = spline(x, y)
    ax_.plot(np.linspace(0, 1, 300), s(np.linspace(0, 1, 300)), '-k', label = 'Data')
    ax_.plot(np.linspace(0, 1, 300), sine_fit(np.linspace(0, 1, 300), *Params), '--r', label = 'Sine fit')
    ax_.legend()
    ax_.set_xlabel('Time [Period]')
    ax_.set_ylabel('Population')

    plt.show()
    
def sine_fit(x, omega, A, delta, C):
    return A*np.sin(omega*x+delta) + C

if __name__ == '__main__':
    main()
