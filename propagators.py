from scipy.fft import fft, ifft
import numpy as np
from time import sleep
from cft import Transformer


def lorentz_fft(data: np.ndarray, resolution=1/32):
    """No periodic time"""
    return Transformer(data, data_intervals=(resolution,)*data.ndim).cft(0).icft(*range(1, data.ndim)).data
def lorentz_ifft(data: np.ndarray, resolution=1/32):
    return Transformer(data, data_intervals=(resolution,)*data.ndim).icft(0).cft(*range(1, data.ndim)).data

def propagate(data: np.ndarray, resolution):
    print('Propagating...')
    data = lorentz_fft(data, resolution=resolution)
    return lorentz_ifft(data, resolution=resolution)



def pt_lorentz_fft(data: np.ndarray, resolution=1/32):
    """Periodic time"""
    return Transformer(fft(data, axis=0), data_intervals=(resolution,)*data.ndim).icft(*range(1, data.ndim)).data
def pt_lorentz_ifft(data: np.ndarray, resolution=1/32):
    return Transformer(ifft(data, axis=0), data_intervals=(resolution,)*data.ndim).cft(*range(1, data.ndim)).data
"""
def lorentz_fft(data: np.ndarray, resolution=1):
    return Transformer(data, data_interval=resolution).cft(0).icft(*range(1, data.ndim)).data
def lorentz_ifft(data: np.ndarray, resolution=1):
    return Transformer(data, data_interval=resolution).icft(0).cft(*range(1, data.ndim)).data
"""

def sg_propagate(current: np.ndarray, center, resolution=1/32):
    print('Propagating...')
    current = lorentz_fft(current, resolution=resolution)
    print('Setting up divk2...')
    for (point,val) in np.ndenumerate(current):
        point_modded = point - center
        current[point] = val / (point_modded[0]**2 - np.dot(point_modded[1:], point_modded[1:]))
    return lorentz_ifft(np.nan_to_num(current, nan=0., posinf=10000, neginf=-10000), resolution=resolution)
