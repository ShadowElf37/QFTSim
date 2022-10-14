from scipy.fft import fftn as spfft, ifftn as spifft, fft as spfft1, ifft as spifft1
import scipy as sp
import numpy as np
from scipy.interpolate import interpn
import threading
from time import sleep

def lorentz_fft(data: np.ndarray):
    return spifft(spfft1(data, axis=0), axes=range(1, data.ndim))
def lorentz_ifft(data: np.ndarray):
    return spfft(spifft1(data, axis=0), axes=range(1, data.ndim))


def p(s):
    while True:
        print(I, '/', s)
        sleep(1)

I = 0

def propagate(current: np.ndarray):
    global I
    print('Interpolating current...')

    for (point,_) in np.ndenumerate(current):
        current[point] /= point[0]**2 - np.dot(point[1:], point[1:])
    current = lorentz_fft(current)
    print('Setting up divk2...')
    print('Propagating...')
    return lorentz_ifft(np.nan_to_num(current, nan=0., posinf=10000, neginf=-10000))
    return field