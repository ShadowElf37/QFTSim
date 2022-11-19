from PIL import Image
import scipy.fft as fft
import numpy as np
import matplotlib.pyplot as plt

pic= Image.open("9nsgatf9wga81.PNG").convert('RGB')
pic= Image.open("mal2.jpg").convert('RGB')

print(np.array(pic.getdata()).shape)
pix = np.array(pic.getdata()).reshape(pic.size[1], pic.size[0], 3)

fig, (default, p1, p2, p3) = plt.subplots(4,4)

subplots = (*default, *p1, *p2, *p3)

for sp in subplots:
    sp.xaxis.tick_top()

default[0].imshow(pix)
default[1].imshow(pix[:,:,0], cmap='Reds_r')
default[2].imshow(pix[:,:,1], cmap='Greens_r')
default[3].imshow(pix[:,:,2], cmap='Blues_r')

"""
max_edge = int(np.sqrt(max_size))

b_red = f_red.copy()
b_red[max_edge:] = 0
b_red[:,max_edge:] = 0
b_red = np.abs(fft.ifft2(b_red))

b_green = f_green.copy()
b_green[max_edge:] = 0
b_green[:,max_edge:] = 0
b_green = np.abs(fft.ifft2(b_green))

b_blue = f_blue.copy()
b_blue[max_edge:] = 0
b_blue[:,max_edge:] = 0
b_blue = np.abs(fft.ifft2(b_blue))
"""

precision = np.csingle

def compress(img, size):
    f_red = fft.fft2(img[:, :, 0]).astype(precision)
    f_green = fft.fft2(img[:, :, 1]).astype(precision)
    f_blue = fft.fft2(img[:, :, 2]).astype(precision)

    rlower_bound = np.sort(np.abs(f_red).flatten())[-size]
    f_red[np.less(np.abs(f_red), rlower_bound)] = 0

    glower_bound = np.sort(np.abs(f_green).flatten())[-size]
    f_green[np.less(np.abs(f_green), glower_bound)] = 0

    blower_bound = np.sort(np.abs(f_blue).flatten())[-size]
    f_blue[np.less(np.abs(f_blue), blower_bound)] = 0

    r = list(zip(*list(np.nonzero(f_red))))
    r = list(zip(r, (f_red[i] for i in r)))
    g = list(zip(*list(np.nonzero(f_green))))
    g = list(zip(g, (f_green[i] for i in g)))
    b = list(zip(*list(np.nonzero(f_blue))))
    b = list(zip(b, (f_blue[i] for i in b)))
    #print(b)

    return f_red.shape, r,g,b

def decompress(shape, r, g, b):
    reds = np.zeros(shape, dtype=precision)
    blues = np.zeros(shape, dtype=precision)
    greens = np.zeros(shape, dtype=precision)

    for i in r:
        reds[i[0]] = i[1]
    for i in g:
        greens[i[0]] = i[1]
    for i in b:
        blues[i[0]] = i[1]

    c_red = np.abs(fft.ifft2(reds))
    c_green = np.abs(fft.ifft2(greens))
    c_blue = np.abs(fft.ifft2(blues))

    c_pix = np.stack((c_red, c_green, c_blue), axis=2).astype(int)

    return c_pix, c_red, c_green, c_blue


for i,p in enumerate((p1, p2, p3)):
    c_pix, c_red, c_green, c_blue = decompress(*compress(pix, 10**(i+2)))
    p[0].imshow(c_pix)
    p[1].imshow(c_red, cmap='Reds_r')
    p[2].imshow(c_green, cmap='Greens_r')
    p[3].imshow(c_blue, cmap='Blues_r')

plt.show()