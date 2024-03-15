# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 08:46:01 2022

@author: pmxtl3
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from pytranskit.optrans.continuous.radoncdt import RadonCDT
from pytranskit.optrans.continuous.cdt import CDT
from pytranskit.optrans.utils.data_utils import signal_to_pdf
import scipy.ndimage as ndimage

figures_location = './figures/rcdt_tests'

N = 250 # resolution

# meshgrid for image arrays
x = np.linspace(-N/2, N/2, N)
y = np.linspace(-N/2, N/2, N)
x, y = np.meshgrid(x, y)
img = np.zeros([N,N])

# # Create image array for distance function cases
# x,y = np.ogrid[0:N, 0:N]
# img = -np.sqrt((x-125)**2 + (y-125)**2)+50

# Circle radius R
R = 50
img[x**2+y**2 <= R**2] = 1

# Edge detection
grad = np.gradient(img)
gx = grad[0] != 0
gy = grad[1] != 0
img = gx | gy
img = img.astype(float)


# # Two circles
# img[(x+70)**2+(y+0)**2 <= R**2] = 1

# # Disc
# img[x**2+y**2 <= (R-1)**2] = 0
# img[(x)**2+(y)**2 <= (R-1)**2] = 0

# # Square
# img[100:150,100:150] = 1

# # Gaussians
# a = 0.001
# b = 0.001
# etax = 0
# etay = 0

# img = np.exp(-a*(x-etax)**2 -b*(y-etay)**2)# + \
#     # np.exp(-a*(x-etax-80)**2 -b*(y-etay-20)**2)

# Invert Image
img = 1-img  

# Smoothing 
img = ndimage.gaussian_filter(img, sigma=1.5, order=0)
img = img*(1/img.max())

# Must be PDF type for RCDT
norm = np.sum(img)
img_norm = img/norm

lims = [np.min(img), np.max(img)] # image value limits
nlims = [np.min(img_norm), np.max(img_norm)] # normalised image limits

# %% Radon and iRadon only

rd = radon(img, circle=False)
ird = iradon(rd, circle=False)

# Relative L2-norm error
err_rd = np.linalg.norm(ird - img) / np.linalg.norm(img)

# Plots and comparison
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.colorbar()
plt.clim(lims)
plt.title('Image')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(1,3,2)
plt.imshow(ird)
plt.colorbar()
plt.clim(lims)
plt.title('Radon-iRadon')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(1,3,3)
plt.imshow(img-ird)
plt.colorbar()
plt.title('$L_2$ Error: {:.3e}'.format(err_rd))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.tight_layout()

# %% CDT and iCDT only

cdt = CDT() # create CDT object
ref = np.ones_like(rd) # Reference function

# Arrays for results
Icdt = []
Iicdt = []

# Perform CDT on each radon transform theta column
for k in range(len(rd[0,:])):
    ref[0,k] = 0 # set first value of reference as 0 for pdf
    # Create pdfs from reference, image
    j0 = signal_to_pdf(ref[:,k])    
    j1 = signal_to_pdf(rd[:,k])
    # Arrays for ranges and interpolation in CDT
    x0 = np.linspace(0,1,len(j0))
    x1 = np.linspace(lims[0], lims[1], len(j1))
    # Perform CDT and inverse
    Icdt_temp,_,_ = cdt.forward(x0, j0, x1, j1)
    Iicdt_temp = cdt.inverse(Icdt_temp, j0, x0)
    Icdt.append(Icdt_temp)
    Iicdt.append(Iicdt_temp)

# Back into original image range domain
Iicdt = np.transpose(Iicdt)
Iicdt *= norm

# iradon for comparison of CDT effects in physical space
orig_phy = iradon(rd, circle=False)
cdt_phy = iradon(Iicdt, circle=False)

# Relative L2-norm error
err_cdt = np.linalg.norm(Iicdt - rd) / np.linalg.norm(rd)
err_cdt_phy =  np.linalg.norm(cdt_phy - orig_phy) / np.linalg.norm(orig_phy)

# Plots and comparison, including iradon of each for comparison in physical space
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.imshow(rd)
plt.colorbar()
plt.title('Image')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(2,3,2)
plt.imshow(Iicdt)
plt.colorbar()
plt.clim(np.min(rd), np.max(rd))
plt.title('CDT-iCDT')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(2,3,3)
plt.imshow(rd-Iicdt)
plt.colorbar()
plt.title('$L_2$ Error: {:.3e}'.format(err_cdt))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(2,3,4)
plt.imshow(orig_phy)
plt.colorbar()
plt.clim(lims)
plt.title('iRadon(Image)')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(2,3,5)
plt.imshow(cdt_phy)
plt.colorbar()
plt.clim(lims)
plt.title('iRadon(CDT-iCDT)')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(2,3,6)
plt.imshow(cdt_phy - orig_phy)
plt.colorbar()
plt.title('$L_2$ Error: {:.3e}'.format(err_cdt_phy))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.tight_layout()

# %% RCDT and inverse RCDT

rcdt = RadonCDT() # create RCDT object
temp = np.ones_like(img) # reference signal

# Perform RCDT, iRCDT and re-normalisation
Ihat = rcdt.forward([0,1], temp, nlims, img_norm)
Irev = rcdt.inverse(Ihat, temp, nlims)
Irev *= norm

# Relative L2-norm error
err_rcdt = np.linalg.norm(Irev - img) / np.linalg.norm(img)

# Plots and comparison
fig = plt.figure(figsize=(13,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.colorbar(orientation='horizontal')
plt.clim(lims)
#plt.title('Image')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(1,3,2)
plt.imshow(Irev)
plt.colorbar(orientation='horizontal')
plt.clim(lims)
#plt.title('RCDT-iRCDT{Image}')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(1,3,3)
plt.imshow(img-Irev)
plt.colorbar(orientation='horizontal', ticks=[-0.5,0,0.5,1.0])
plt.title('$L_2$ Error: {:.3e}'.format(err_rcdt))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.tight_layout()
