from scipy.fft import fftn as spfft, ifftn as spifft
from scipy.spatial.distance import minkowski
import numpy as np

def lorentz_fft(data: np.ndarray):
    return spfft(spifft(data, axes=(0,)), axes=range(1, data.ndim))
def lorentz_ifft(data: np.ndarray):
    return spifft(spfft(data, axes=(0,)), axes=range(1, data.ndim))

def propagate(current: np.ndarray):
    divk2 = np.ones_like(current)
    for (point,_) in np.ndenumerate(divk2):
        divk2[point] /= np.clip(point[0]**2 - sum(point[i]**2 for i in range(len(point)-1)), 0.1, None)
    return lorentz_ifft(np.multiply(lorentz_fft(current), divk2))