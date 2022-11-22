import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#box phi = 0
#d_t^2 - d_x^2 phi = 0
#phi(t+2, x) - 2phi(t+1, x) = phi(t, x+2) - 2phi(t, x+1)
# phi(t+1, x) = phi(t, x+1) + phi(t, x) - phi(t, x)

# t2 - 2t1 = k
# t3 - 2t2 = l
#


TDIM = 50
XDIM = 100

field = np.zeros((TDIM+2, XDIM+2), np.complex64)

field[0,:] = np.exp((-np.arange(0, XDIM)-10)**2)

field[2, 0] - 2*field[1, 0] =

t_step_vector = np.array([field[0, x+2] - 2*field[0, x+1] for x in range(XDIM)])
t_step_matrix =

