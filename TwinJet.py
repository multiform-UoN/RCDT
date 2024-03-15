# %%
# IMPORTS

from time import time
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp
from pytranskit.optrans.continuous.radoncdt import RadonCDT
# Database module
from ezyrb import Database
# Dimensionality reduction methods
from ezyrb import POD
# Approximation/interpolation methods
from ezyrb import GPR, RBF, linear
# Model order reduction class
from ezyrb import ReducedOrderModel as ROM

# FFT for paper
from scipy.fftpack import fft2, ifft2

# %%
# Load datasets
datapath = "../Data/2prop/XVelocity/"

images = np.array([np.load(datapath+"XVelocity"+str(i)+".npy") for i in range(26)])
Nt = int(images.shape[0])
ind = range(0,Nt)
params = np.linspace(0,Nt,Nt)
# No. of uniform grid points for griddata interp
Ny = int(images.shape[1])
Nz = int(images.shape[2])
# %%
# reduce/form to training set
# Setting training data (a reduced version of the full dataset)
skip = 10
ind_tr = ind[::skip]
Ntr = len(ind_tr)
images_tr = images[ind_tr,:,:]

# POD and interpolation parameters
Nmodes = 100 # Number of modes to use for ROMs
k_tr = math.ceil(Ntr/5)# reference index to use to check training set (< Ntr)
k = math.ceil(k_tr*skip + skip*0.5) # reference index to use to  predict  (< Nt)
# snapshots either side of target to use for interp
k0 = math.floor(k/skip)
k1 = math.ceil(k/skip)
k0c = k/skip - k0  # interpolation coefficient

# vectors to pass to Ezyrb Databased
indtr = np.array(list(ind_tr)).reshape(-1,1)
snaps = np.array([images_tr[i,:,:].reshape(Nz*Ny) for i in range(Ntr)])
# %%
# Physical ROM using EZyRB

'''
Perform POD on data in physical space.
'''

tic = time()

# Physical ROM construction
db = Database(indtr, snaps)
phys_pod = POD(rank = Nmodes)

rom = ROM(db, phys_pod, linear.Linear())
rom.fit();

# POD singular values
s_phys = phys_pod.singular_values

toc = time()
print("Physical ROM time elapsed: {:.2f} minutes".format((toc-tic)/60))
# %%
# TEST Physical ROM

# # Manual interpolation
# p = (alpha_tr[k0]*k0c + alpha_tr[k1]*(1-k0c)).T
# Interpolation using EZYRB
p = rom.predict(k).reshape(Ny,Nz)
# if thresholding:
#     p = np.where(p > 0.5, 1, 0)

fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(images[k,:,:].T)
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
#plt.clim(0,1)

plt.subplot(1,3,2)
plt.imshow(p.T)
plt.colorbar(orientation='horizontal')
plt.clim(np.min(images[k]),np.max(images[k]))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(images[k,:,:].T-p.T)))
plt.colorbar(orientation='horizontal')
plt.clim(-2,0)
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((images[k,:,:]-p),1)/np.linalg.norm(images[k],1)))
plt.tight_layout()


# %%
# FFT space ROM (no-padding case only)

# Compute FFT of the data
images_fft = np.array([fft2(images_tr[j]) for j in range(Ntr)])
images_fft = np.array([[images_fft[j].real,images_fft[j].imag] for j in range(Ntr)])

db_fft = Database(indtr, np.reshape(images_fft, (Ntr, -1)))
pod_fft = POD(rank = Nmodes)
rom_fft = ROM(db_fft, pod_fft, linear.Linear())
rom_fft.fit();

s_fft = pod_fft.singular_values

# %%
# TEST FFT ROM

# # Manual interpolation
# p = (alpha_fft[k0]*k0c + alpha_fft[k1]*(1-k0c))

# # Interpolation using EZYRB
p = rom_fft.predict(k).reshape(2,Ny,Nz)

p = p[0,:,:]+1j*p[1,:,:]
p = ifft2(p).real.T

# if thresholding:
#     p = np.where(p > 0.5, 1, 0)
fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(images[k,:,:].T)
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
# plt.clim(0,1)

plt.subplot(1,3,2)
plt.imshow(p)
plt.colorbar(orientation='horizontal')
plt.clim(np.min(images[k]),np.max(images[k]))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(images[k,:,:].T-p)))
plt.colorbar(orientation='horizontal')
plt.clim(-2,0)#(-3,1)
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm(abs(images[k,:,:].T-p),1)/np.linalg.norm(images[k],1)))
plt.tight_layout()

# %%
# RCDT - iRCDT
img_tr_pos = images_tr*(images_tr>0)
img_tr_neg = images_tr*(images_tr<0)*(-1)
# get normalising constants
norm_pos = np.sum(img_tr_pos, (1,2))
norm_neg = np.sum(img_tr_neg, (1,2))

# Normalize
for i in range(Ntr):
    img_tr_pos[i] /= norm_pos[i]
    img_tr_neg[i] /= norm_neg[i]

rcdt = RadonCDT(theta=np.linspace(0,180,360)) # create RCDT object
temp = np.ones_like(images_tr[0].T) # reference signal
Irev_temp = np.zeros(images_tr.shape[0])

tic = time()

# Get shape for RCDT array
rcdt_shape = (Ntr,) + np.shape(rcdt.forward([0,1], temp, [0,1], temp))

# Arrays for RCDT calculations
IhatPos = np.zeros(rcdt_shape)
IhatNeg = np.zeros(rcdt_shape)
Irev = np.zeros_like(images_tr)
err_rcdt = np.zeros(Ntr)

# Perform RCDT, iRCDT and re-normalisation
for i in range(Ntr):
    IhatPos[i] = rcdt.forward([0,1], temp, [0,1],img_tr_pos[i].T)
    IhatNeg[i] = rcdt.forward([0,1], temp, [0,1],img_tr_neg[i].T)
    
    Irev_temp = norm_pos[i]*rcdt.inverse(IhatPos[i], temp, [0,1]).T - norm_neg[i]*rcdt.inverse(IhatNeg[i], temp, [0,1]).T
    Irev[i] = Irev_temp
    
    #L2-norm error
    err_rcdt[i] = np.linalg.norm(Irev[i] - images_tr[i]) / np.linalg.norm(images_tr[i])
    
    
toc = time()
print("RCDT-iRCDT time elapsed: {:.2f} minutes".format((toc-tic)/60))

# %% RCDT POD

'''
Perform POD on data in Radon-CDT space.
'''

tic = time()
# Construct database object, in RCDT space, for ROM(POD) use
db = Database(indtr, np.reshape(np.stack([IhatPos,IhatNeg],-1), (Ntr, -1)))
# Construct POD object; default, singular val. decomp (SVD); for Nmodes rank
rcdt_pod = POD(rank = Nmodes)
# Perform reduce order modelling (ROM) i.e. POD, GPR for prediction/approx.
rom_rcdt = ROM(db, rcdt_pod, RBF())
rom_rcdt.fit();
# Retrieve singular vals. of decomp
s_rcdt = rcdt_pod.singular_values

toc = time()
print("RCDT ROM time elapsed: {:.2f} minutes".format((toc-tic)/60))

# %%
# TEST RCDT inversion only

fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(images_tr[k_tr,:,:].T)
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
#plt.title('Image')

plt.subplot(1,3,2)
plt.imshow(Irev[k_tr,:,:].T)
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.clim(np.min(images_tr[k_tr,:,:]),np.max(images_tr[k_tr,:,:]))
#plt.title('RCDT-iRCDT{Image}')

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(images_tr[k_tr,:,:].T-Irev[k_tr,:,:].T)))
plt.colorbar(orientation='horizontal')
plt.clim(-2,0)
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.title('$L_2$ Error: {:.3e}'.format(err_rcdt[k_tr]))

plt.tight_layout()

# %%
# TEST RCDT interpolation

# Manual interpolation
# Ihat_interp = (Ihat[k0]*k0c + Ihat[k1]*(1-k0c))

# # Interpolation using EZYRB
Ihat_interp = rom_rcdt.predict(k).reshape(rcdt_shape[1:] + (2,))

Irev_interp = rcdt.inverse(Ihat_interp[...,0], temp, [0,1])
Irev_interp_neg = rcdt.inverse(Ihat_interp[...,1], temp, [0,1])
# Re-normalise, using linear interpd norm
Irev_interp *= (norm_pos[k0]*k0c + norm_pos[k1]*(1-k0c))
Irev_interp_neg *= (norm_neg[k0]*k0c + norm_neg[k1]*(1-k0c))

p = Irev_interp - Irev_interp_neg

# L2-norm error
err_interp = np.linalg.norm(p - images[k].T) / np.linalg.norm(images[k])

plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.imshow(images[k,:,:].T)
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
#plt.title('Image')

plt.subplot(1,3,2)
plt.imshow(p)
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.clim(np.min(images[k,:,:]),np.max(images[k]))
#plt.title('RCDT interpolation')

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(images[k,:,:].T-p)))
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.clim(-2,0)
plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((images[k,:,:]-p.T),1)/np.linalg.norm(images[k],1)))
# plt.title('$L_2$ Error: {:.3e}'.format(err_interp))

plt.tight_layout()
# %%

