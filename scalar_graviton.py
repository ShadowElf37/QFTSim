import numpy as np
import scipy as sp
import fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# box h - Lh^2 - J = 0

TDIM = 50
XDIM = 100
YDIM = 100

CENTER = np.array((TDIM//2, XDIM//2, YDIM//2))

J = 1

fieldJ = np.zeros((TDIM, XDIM, YDIM), np.complex64)

X = np.arange(0, XDIM)
Y = np.arange(0, YDIM)
#T = np.arange(0, TDIM)

speed = 1
radius = 5
rotation_radius = 20
#mask = lambda t: (((X[np.newaxis,:]-CENTER[1]-rotation_radius*np.cos(t*2*np.pi/TDIM))**2 + (Y[:,np.newaxis]-CENTER[2]-rotation_radius*np.sin(t*2*np.pi/TDIM))**2) < radius**2) if t < 50 else 0\
mask = lambda t: (((X[np.newaxis,:]-CENTER[1])**2 + (Y[:,np.newaxis]-CENTER[2])**2) < radius**2) if t < 50 else 0
#fieldJ[:,XDIM//2,YDIM//2] = np.ones_like(fieldJ[:,XDIM//2,YDIM//2])*J
for t in range(TDIM):
    fieldJ[t][mask(t)] = J

#plt.imshow(np.abs(fieldJ[20]))
#plt.show()

res = 0.1

field0 = fft.propagate(fieldJ, CENTER, resolution=res)
field = field0 + 0.5*fft.propagate(field0**2, CENTER, resolution=res)


real = np.abs(field)**2

#rbf = sp.interpolate.Rbf(xi, yi, t0, function='linear')
#ai = rbf(xi, yi)



fig = plt.figure(figsize=(8,8))
image = plt.imshow(real[0], vmax=np.max(real), origin='lower', interpolation='bilinear')#, extent=[-XDIM/2, XDIM/2, -YDIM/2, YDIM/2])
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