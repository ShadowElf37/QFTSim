import numpy as np
import scipy as sp
import propagators
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import pi
from random import randint

# box h - Lh^2 - J = 0

TDIM = 50
XDIM = 50
YDIM = 50

CENTER = np.array((TDIM//2, XDIM//2, YDIM//2))

J = 10000
field0 = np.zeros((TDIM, XDIM, YDIM), np.complex64)

X = np.arange(0, XDIM)
Y = np.arange(0, YDIM)
X,Y = np.meshgrid(X, Y)
#T = np.arange(0, TDIM)
#field0[:, XDIM//2, :] = 1
#field0[0] = sum(np.exp(-((X-randint(0, 50))**2/10 + (Y-randint(0, 50))**2/10)) for _ in range(5))*J
field0[:, randint(0, XDIM), randint(0, YDIM)] = J
init = field0[0].copy()
#field0[1] = np.exp(-(X-21)**2)

#plt.imshow(np.abs(field0[0]))
#plt.show()
#exit()

speed = 1
radius = 3
rotation_radius = 10
#mask = lambda t: (((X[np.newaxis,:]-CENTER[1]-rotation_radius*np.cos(t*2*np.pi/TDIM*speed))**2 + (Y[:,np.newaxis]-CENTER[2]-rotation_radius*np.sin(t*2*np.pi/TDIM*speed))**2) < radius**2)\
#mask = lambda t: (((X[np.newaxis,:]-t)**2 + (Y[:,np.newaxis]-t)**2) <= radius**2)
#fieldJ[:,XDIM//2,YDIM//2] = np.ones_like(fieldJ[:,XDIM//2,YDIM//2])*J
#for t in range(TDIM):
#    fieldJ[t,mask(t)] = J

#plt.imshow(np.abs(fieldJ[20]))
#plt.show()

res = 1/(pi)**2
# I have no idea why the resolution needs to be this

field0 = propagators.sg_propagate(field0, CENTER, resolution=res)
field = field0#+ 0.5*propagators.sg_propagate(field0**2, CENTER, resolution=res)


real = np.abs(field)**2

#rbf = sp.interpolate.Rbf(xi, yi, t0, function='linear')
#ai = rbf(xi, yi)

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
#p, = ax.plot(X, real[0,:])

image = ax1.imshow(real[0], origin='lower')#, interpolation='bilinear')#, extent=[-XDIM/2, XDIM/2, -YDIM/2, YDIM/2])
static = ax2.imshow(np.abs(init), origin='lower')
static1 = ax3.imshow(real[0], origin='lower')

#fig.colorbar(image)

def animate_func(i):
    image.set_array(real[i])
    #p.set_ydata(real[i,:])
    print(i)
    return image,

anim = animation.FuncAnimation(
                               fig,
                               animate_func,
                               frames = TDIM,
                               interval = 50, # in ms
                               )

plt.xlabel('X')
#plt.ylabel('Y')

plt.show()


#scipy.fft.

#field1 =