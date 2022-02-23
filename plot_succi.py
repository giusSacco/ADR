#! /usr/bin/env python3

import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import sys, os
import re
from scipy.fft import fft, fftfreq
from scipy.interpolate import InterpolatedUnivariateSpline as spline
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

        ax2.plot(o/(6.28), a*100, 'ok')
        ax2.set_xscale('log')
        ax2.set_ylabel(r'Modulation Amplitude [\%]')

        ax3.plot(o, p, 'ok')
        ax3.set_xscale('log')
        ax3.set_ylabel('Relative Phase[°]')
        ax3.set_ylim([-100, 100])

        ax3.set_xlabel(r'$\frac{\omega  a}{2 \pi}$')

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
    
    omegas, Img, maxes, peaks, spectra = zip(*sorted(zip(omegas, Img, maxes, peaks, spectra), reverse=False))

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

    plt.show()
    

if __name__ == '__main__':
    main()
