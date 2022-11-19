import numpy as np
import scipy as sp
import fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# box + m^2 phi - g/2 phi^2 = 0

XDIM = 10
DX = 0.01
TDIM = 10
DT = 0.1

X = np.arange(-XDIM/2, XDIM/2, DX, dtype=np.complex64)
T = np.arange(-TDIM/2, TDIM/2, DT, dtype=np.complex64)
field = np.zeros((T.size, X.size), np.complex64)
field[0] = np.exp(-(X-20)**2)

res = 0.1

field = fft.propagate(field, CENTER, resolution=res)
field += 0.5*fft.propagate(field**2, CENTER, resolution=res)


real = np.abs(field)**2

#rbf = sp.interpolate.Rbf(xi, yi, t0, function='linear')
#ai = rbf(xi, yi)



plt.plot(real[0], vmax=np.max(real), origin='lower', interpolation='bilinear')#, extent=[-XDIM/2, XDIM/2, -YDIM/2, YDIM/2])
#plt.scatter(xi, yi, c=t0)
plt.colorbar()



def animate_func(i):
    image.set_array(real[i][:][:])
    print(i)
    return [image]

anim = animation.FuncAnimation(
                               fig,
                               animate_func,
                               frames = TDIM,
                               interval = 50, # in ms
                               )

plt.xlabel('X')
plt.ylabel('Y')

plt.show()


#scipy.fft.

#field1 =