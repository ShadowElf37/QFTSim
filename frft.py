# Fractional Fourier Transform implementation
# See http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/fourint.pdf
# Bailey, Swartztrauber "A Fast Method for the Numerical Evaluation of Continuous Fourier and Laplace Transforms"

from numpy import pi, exp
import numpy as np
import scipy as sp
from scipy.fft import fftn, ifftn, fft, ifft
import matplotlib.pyplot as plt

ipi = pi*1j

def frft(x: np.ndarray, alpha: float):
    m = x.size
    y = np.concatenate((np.array([xi*exp(-ipi*i**2*alpha) for (i,),xi in np.ndenumerate(x)]), np.zeros(m)))
    z = np.concatenate((np.array([exp(ipi * i ** 2 * alpha) for i in range(m)]), (np.array([exp(ipi * (i-m) ** 2 * alpha) for i in range(m)]))))
    transform = ifft(np.multiply(fft(y),fft(z)))
    return np.array([exp(-ipi*k**2*alpha) * transform[k] for k in range(m)])

def cft(x: np.ndarray, resolution):
    m = x.size
    beta = gamma = resolution
    delta = beta*gamma/(2*pi)
    transform = frft(np.array([xi*exp(ipi*i*m*delta) for (i,),xi in np.ndenumerate(x)]), delta)
    return beta * np.array([exp(ipi*(k-m/2)*m*delta)*transform[k] for k in range(m)])

# inverse - just negative delta
def icft(x: np.ndarray, resolution):
    m = x.size
    beta = gamma = resolution
    delta = -beta*gamma/(2*pi)
    transform = frft(np.array([xi*exp(ipi*i*m*delta) for (i,),xi in np.ndenumerate(x)]), delta)
    return beta * np.array([exp(ipi*(k-m/2)*m*delta)*transform[k] for k in range(m)])

if __name__ == "__main__":
    resolution = 0.3#np.sqrt(2*pi)/256
    X = np.arange(-10,10,resolution)
    y1 = 1/np.sqrt(2*pi)*exp(-X**2/2)
    y2 = cft(y1, resolution)

    plt.plot(X, y1, color='green') # original function
    plt.plot(X, y2, color='blue', linewidth=3) # computed fourier transform
    plt.plot(X, exp(-X**2/2), color='orange', linewidth=1) # actual fourier transform
    plt.show()