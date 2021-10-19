import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def logistica(x, kappa, a, c, d):
    return kappa/(1+d*np.exp(a*(x-c)))

dt = 0.01

dirname = 'Arrays_seed2'

def a_eff(filename : str):

    y = np.loadtxt(filename)

    x = np.arange(len(y))*dt

    params, _ = curve_fit(logistica, x, y)

    return params[0]

winds = [0, 0.15, 0.3]

a = [] 

for f in ['constwind_0.0001.txt', 'constwind_0.15.txt', 'constwind_0.3.txt']:
    a_tmp = a_eff(os.path.join(dirname, f))
    a.append(a_tmp)

print(a)