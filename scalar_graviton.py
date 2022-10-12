import numpy as np
import scipy as sp
import fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# box h - Lh^2 - J = 0

TDIM = 100
XDIM = 200
YDIM = 200

J = 100

fieldJ = np.zeros((TDIM, XDIM, YDIM), np.complex128)

X = np.arange(0, XDIM)
Y = np.arange(0, YDIM)
#T = np.arange(0, TDIM)

speed = 1
radius = 5
rotation_radius = 20
mask = lambda t: (X[np.newaxis,:]-XDIM/2-rotation_radius*np.cos(t*2*np.pi/TDIM*speed))**2 + (Y[:,np.newaxis]-YDIM/2-rotation_radius*np.sin(t*2*np.pi/TDIM*speed))**2 < radius**2
for t in range(TDIM):
    fieldJ[t][mask(t)] = J
print(fieldJ[0])
print(fieldJ.shape)

field0 = fft.propagate(fieldJ)
field = field0 + 0.1*fft.propagate(field0**2)


xi, yi = np.linspace(-XDIM/2, XDIM/2, 50), np.linspace(-YDIM/2, YDIM/2, 50)
xi, yi = np.meshgrid(xi, yi)
real = np.abs(field)

#rbf = sp.interpolate.Rbf(xi, yi, t0, function='linear')
#ai = rbf(xi, yi)




fig = plt.figure(figsize=(8,8))
image = plt.imshow(real[0], vmin=0, vmax=np.median(real)*5, origin='lower', interpolation='bilinear')#, extent=[-XDIM/2, XDIM/2, -YDIM/2, YDIM/2])
#plt.scatter(xi, yi, c=t0)
#plt.colorbar()

def animate_func(i):
    image.set_array(real[i])
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