# %%
# Imports

from time import time
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from pytranskit.optrans.continuous.radoncdt import RadonCDT

from skimage.transform import radon
# Database module
from ezyrb import Database
# Dimensionality reduction methods
from ezyrb import POD
# Approximation/interpolation methods
from ezyrb import linear, GPR, RBF
# Model order reduction class
from ezyrb import ReducedOrderModel as ROM
# FFT for paper
from scipy.fftpack import fft2, ifft2

# function to return unique values up to a tolerance
def uniquetol(A, tol=1e-3):
    return A[~(np.triu(np.abs(A[:,None] - A) <= tol,1)).any(0)]

# %%
# READ AND PROCESS DATA

# folder location
figures_location = './figures/multiphase_test'
data_location = '../Data/multiphase/'

# Load in alpha, C, coordinate arrays
alpha_orig = np.load(data_location+'alpha.npy')
C = np.load(data_location+'C.npy')
coords = C.reshape(3, 18750)

# seperate x and y coordinates
x=coords[0]
xi=uniquetol(x)
y=coords[1]
yi=uniquetol(y)

# No. of uniform grid points for griddata interp
Nx = len(xi)
Ny = len(yi)

# No. of timepoints
Nt = alpha_orig.shape[1]
ind = range(0,Nt)

# reshape data
alpha = alpha_orig.reshape(Ny,Nx,Nt)[::-1,:,:].T

# %%
# OPTIONS

# pre-processing image options
subtract_min = True

# thresholding
thresholding = True#False

# Padding - add padding to image arrays before RCDT
pad = False
pad_val = (0,0) # (x,y)
pad_width = (30,30) # (x,y)
pad_type = 'constant'

# Setting training data (a reduced version of the full dataset)
skip = 20
ind_tr = ind[::skip]
Ntr = len(ind_tr)
alpha_tr = alpha[ind_tr,:,:]

# POD and interpolation parameters
Nmodes = 10 # Number of modes to use for ROMs
k_tr = math.ceil(Ntr/5)# reference index (wrt Ntr) to use to check training set
k = k_tr*skip + math.ceil(skip*0.5) # reference index (wrt Nt) to use to  predict
# snapshots either side of target to use for interp
k0 = math.floor(k/skip)
k1 = math.ceil(k/skip)
k0c = k/skip - k0  # interpolation coefficient

# vectors to pass to Ezyrb Databased
indtr = np.array(list(ind_tr)).reshape(-1,1)
snaps = np.array([alpha_tr[i,:,:].reshape(Nx*Ny) for i in range(Ntr)])


# %%
# Pre-processing of inputs 

'''
   Ideas here for potential pre-processing of images/arrays before RCDT
   Thresholding, edge detection?, etc.
'''

# threshold
# alpha = np.where(alpha > 0.5, 1, 0)


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
# rom = ROM(db, phys_pod, RBF())
rom.fit();

# POD singular values
s_phys = phys_pod.singular_values

toc = time()
print("Physical ROM time elapsed: {:.2f} minutes".format((toc-tic)/60))

# %%
# RECONSTRUCTION TEST Physical ROM

# # Manual interpolation
# p = (alpha_tr[k0]*k0c + alpha_tr[k1]*(1-k0c)).T
# Interpolation using EZYRB
p = rom.predict(k_tr*skip).reshape(Nx,Ny).T

fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(alpha[k_tr*skip,:,:].T)
plt.colorbar(orientation='horizontal')
plt.clim(0,1)

plt.subplot(1,3,2)
plt.imshow(p)
plt.colorbar(orientation='horizontal')
#plt.clim(0,1)

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(alpha[k_tr*skip,:,:].T-p)+1e-10))
plt.colorbar(orientation='horizontal')
plt.clim(-2,0)
# plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((alpha[k,:,:].T-p),1)/np.linalg.norm(alpha[k],1)))
plt.tight_layout()

plt.savefig('Multiphase_reconstruction_phys'+str(Nmodes)+'.pdf', format='pdf')


p = np.where(p > 0.5, 1, 0)
fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(alpha[k_tr*skip,:,:].T)
plt.colorbar(orientation='horizontal')
plt.clim(0,1)

plt.subplot(1,3,2)
plt.imshow(p)
plt.colorbar(orientation='horizontal')
#plt.clim(0,1)

plt.subplot(1,3,3)
plt.imshow((abs(alpha[k_tr*skip,:,:].T-p)))
plt.colorbar(orientation='horizontal')
# plt.clim(-2,0)
# plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((alpha[k,:,:].T-p),1)/np.linalg.norm(alpha[k],1)))
plt.tight_layout()

plt.savefig('Multiphase_reconstruction_phys_thres'+str(Nmodes)+'.pdf', format='pdf')

# %%
# INTERPOLATION TEST Physical ROM

# # Manual interpolation
# p = (alpha_tr[k0]*k0c + alpha_tr[k1]*(1-k0c)).T
# Interpolation using EZYRB
p = rom.predict(k).reshape(Nx,Ny).T

fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(alpha[k,:,:].T)
plt.colorbar(orientation='horizontal')
plt.clim(0,1)

plt.subplot(1,3,2)
plt.imshow(p)
plt.colorbar(orientation='horizontal')
#plt.clim(0,1)

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(alpha[k,:,:].T-p)+1e-10))
plt.colorbar(orientation='horizontal')
plt.clim(-2,0)
# plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((alpha[k,:,:].T-p),1)/np.linalg.norm(alpha[k],1)))
plt.tight_layout()

plt.savefig('Multiphase_interpolation_phys'+str(Nmodes)+'.pdf', format='pdf')

p = np.where(p > 0.5, 1, 0)
fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(alpha[k,:,:].T)
plt.colorbar(orientation='horizontal')
plt.clim(0,1)

plt.subplot(1,3,2)
plt.imshow(p)
plt.colorbar(orientation='horizontal')
#plt.clim(0,1)

plt.subplot(1,3,3)
plt.imshow(abs(alpha[k,:,:].T-p))
plt.colorbar(orientation='horizontal')
# plt.clim(-2,0)
# plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((alpha[k,:,:].T-p),1)/np.linalg.norm(alpha[k],1)))
plt.tight_layout()

plt.savefig('Multiphase_interpolation_phys_thres'+str(Nmodes)+'.pdf', format='pdf')

# # %%
# # FFT space ROM (no-padding case only)

# # Compute FFT of the data
# alpha_fft = np.array([fft2(alpha_tr[j]) for j in range(Ntr)])
# alpha_fft = np.array([[alpha_fft[j].real,alpha_fft[j].imag] for j in range(Ntr)])

# db_fft = Database(indtr, np.reshape(alpha_fft, (Ntr, -1)))
# pod_fft = POD(rank = Nmodes)
# rom_fft = ROM(db_fft, pod_fft, linear.Linear())
# rom_fft.fit();

# s_fft = pod_fft.singular_values

# # %%
# # TEST FFT ROM

# # # Manual interpolation
# # p = (alpha_fft[k0]*k0c + alpha_fft[k1]*(1-k0c))

# # # Interpolation using EZYRB
# p = rom_fft.predict(k).reshape(2,Nx,Ny)

# p = p[0,:,:]+1j*p[1,:,:]
# p = ifft2(p).real.T

# if thresholding:
#     p = np.where(p > 0.5, 1, 0)
# fig = plt.figure(figsize=(15,5))

# plt.subplot(1,3,1)
# plt.imshow(alpha[k,:,:].T)
# plt.colorbar(orientation='horizontal')
# plt.clim(0,1)

# plt.subplot(1,3,2)
# plt.imshow(p)
# plt.colorbar(orientation='horizontal')
# plt.clim(0,1)

# plt.subplot(1,3,3)
# plt.imshow(np.log10(abs(alpha[k,:,:].T-p))+1e-10)
# plt.colorbar(orientation='horizontal')
# plt.clim(-2,0)#(-3,1)
# # plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm(abs(alpha[k,:,:].T-p),1)/np.linalg.norm(alpha[k],1)))
# plt.tight_layout()

# %%
# RCDT - iRCDT

# subtract the minimum across all training images to reduce boundary effects
minalpha = np.min(alpha_tr,0)*subtract_min
alpha_tr_rcdt = alpha_tr - minalpha

# get normalising constants
norm_alpha = np.sum(alpha_tr_rcdt, (1,2))
alpha_tr_rcdt = alpha_tr_rcdt / norm_alpha[:,None,None]

rcdt = RadonCDT(theta=np.linspace(0,180,360)) # create RCDT object

if pad:
    temp = np.ones_like(alpha_pad[0]) # reference signal
    Irev_temp = np.zeros(alpha_pad.shape[0])

else:
    temp = np.ones_like(alpha_tr_rcdt[0].T) # reference signal
    Irev_temp = np.zeros(alpha_tr_rcdt.shape[0])

tic = time()

# Get shape for RCDT array
rcdt_shape = (Ntr,) + np.shape(rcdt.forward([0,1], temp, [0,1], temp))

# Arrays for RCDT calculations
Ihat = np.zeros(rcdt_shape)
Irev = np.zeros_like(alpha_tr_rcdt)
err_rcdt = np.zeros(Ntr)

# Perform RCDT, iRCDT and re-normalisation
for i in range(Ntr):
    if pad:
        Ihat[i] = rcdt.forward([0,1], temp, [0,1], alpha_pad[i])
    else:
        Ihat[i] = rcdt.forward([0,1], temp, [0,1], alpha_tr_rcdt[i].T)
        
    Irev_temp = rcdt.inverse(Ihat[i], temp, [0,1]).T
    Irev_temp *= norm_alpha[i]
    Irev_temp += minalpha[i]
    
    if pad:
        Irev[i] = Irev_temp[pad_width[0]:-pad_width[0], pad_width[1]:-pad_width[1]]
    else:
        Irev[i] = Irev_temp
    
    #L2-norm error
    err_rcdt[i] = np.linalg.norm(Irev[i] - alpha_tr[i]) / np.linalg.norm(alpha_tr[i])
    
    
toc = time()
print("RCDT-iRCDT time elapsed: {:.2f} minutes".format((toc-tic)/60))

# %% RCDT POD

'''
Perform POD on data in Radon-CDT space.
'''

tic = time()
# Construct database object, in RCDT space, for ROM(POD) use
db = Database(indtr, np.reshape(Ihat, (Ntr, -1)))
# Construct POD object; default, singular val. decomp (SVD); for Nmodes rank
rcdt_pod = POD(rank = Nmodes)
# Perform reduce order modelling (ROM) i.e. POD, GPR for prediction/approx.
rom_rcdt = ROM(db, rcdt_pod, linear.Linear())
# rom_rcdt = ROM(db, rcdt_pod, RBF())
rom_rcdt.fit();
# Retrieve singular vals. of decomp
s_rcdt = rcdt_pod.singular_values

toc = time()
print("RCDT ROM time elapsed: {:.2f} minutes".format((toc-tic)/60))



# %%
# TEST RCDT reconstruction

# Manual interpolation
# Ihat_interp = (Ihat[k0]*k0c + Ihat[k1]*(1-k0c))

# # Interpolation using EZYRB
Ihat_interp = rom_rcdt.predict(k_tr*skip).reshape(rcdt_shape[1:])

Irev_interp = rcdt.inverse(Ihat_interp, temp, [0,1])
Irev_interp *= (norm_alpha[k_tr])
Irev_interp += minalpha.T

if pad:
    Irev_interp = Irev_interp[pad_width[0]:-pad_width[0], pad_width[1]:-pad_width[1]]

p = Irev_interp

# L2-norm error
err_interp = np.linalg.norm(Irev_interp - alpha[k_tr*skip].T) / np.linalg.norm(alpha[k_tr*skip])

plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.imshow(alpha[k_tr*skip,:,:].T)
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
#plt.title('Image')

plt.subplot(1,3,2)
plt.imshow(p[:,:])
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
plt.clim(0,1)
#plt.title('RCDT interpolation')

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(alpha[k_tr*skip,:,:].T-p[:,:])))
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
plt.clim(-2,0)
# plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((alpha[k,:,:].T-p),1)/np.linalg.norm(alpha[k],1)))
# plt.title('$L_2$ Error: {:.3e}'.format(err_interp))

plt.tight_layout()

plt.savefig('Multiphase_reconstruction_rcdt'+str(Nmodes)+'.pdf', format='pdf')

p = np.where(p > 0.5, 1, 0)

# L2-norm error
err_interp = np.linalg.norm(Irev_interp - alpha[k_tr*skip].T) / np.linalg.norm(alpha[k_tr*skip])

plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.imshow(alpha[k_tr*skip,:,:].T)
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
#plt.title('Image')

plt.subplot(1,3,2)
plt.imshow(p[:,:])
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
plt.clim(0,1)
#plt.title('RCDT interpolation')

plt.subplot(1,3,3)
plt.imshow((abs(alpha[k_tr*skip,:,:].T-p[:,:])))
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
# plt.clim(-2,0)
# plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((alpha[k,:,:].T-p),1)/np.linalg.norm(alpha[k],1)))
# plt.title('$L_2$ Error: {:.3e}'.format(err_interp))

plt.tight_layout()

plt.savefig('Multiphase_reconstruction_rcdt_thres'+str(Nmodes)+'.pdf', format='pdf')


# %%
# TEST RCDT interpolation

# Manual interpolation
# Ihat_interp = (Ihat[k0]*k0c + Ihat[k1]*(1-k0c))

# # Interpolation using EZYRB
Ihat_interp = rom_rcdt.predict(k).reshape(rcdt_shape[1:])

Irev_interp = rcdt.inverse(Ihat_interp, temp, [0,1])
Irev_interp *= (norm_alpha[k0]*k0c + norm_alpha[k1]*(1-k0c))
Irev_interp += minalpha.T

if pad:
    Irev_interp = Irev_interp[pad_width[0]:-pad_width[0], pad_width[1]:-pad_width[1]]

p = Irev_interp

# L2-norm error
err_interp = np.linalg.norm(Irev_interp - alpha[k].T) / np.linalg.norm(alpha[k])

plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.imshow(alpha[k,:,:].T)
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
#plt.title('Image')

plt.subplot(1,3,2)
plt.imshow(p[:,:])
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
plt.clim(0,1)
#plt.title('RCDT interpolation')

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(alpha[k,:,:].T-p[:,:])))
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
plt.clim(-2,0)
# plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((alpha[k,:,:].T-p),1)/np.linalg.norm(alpha[k],1)))
# plt.title('$L_2$ Error: {:.3e}'.format(err_interp))

plt.tight_layout()

plt.savefig('Multiphase_interpolation_rcdt'+str(Nmodes)+'.pdf', format='pdf')

p = np.where(p > 0.5, 1, 0)

# L2-norm error
err_interp = np.linalg.norm(Irev_interp - alpha[k].T) / np.linalg.norm(alpha[k])

plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.imshow(alpha[k,:,:].T)
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
#plt.title('Image')

plt.subplot(1,3,2)
plt.imshow(p[:,:])
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
plt.clim(0,1)
#plt.title('RCDT interpolation')

plt.subplot(1,3,3)
plt.imshow((abs(alpha[k,:,:].T-p[:,:])))
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
# plt.clim(-2,0)
# plt.title('$L_1$ Error: {:.3e}'.format(np.linalg.norm((alpha[k,:,:].T-p),1)/np.linalg.norm(alpha[k],1)))
# plt.title('$L_2$ Error: {:.3e}'.format(err_interp))

plt.tight_layout()

plt.savefig('Multiphase_interpolation_rcdt_thres'+str(Nmodes)+'.pdf', format='pdf')

# %% POD mode comparison

# Singular values
s_rcdt = rcdt_pod.singular_values.T
s_phy = phys_pod.singular_values.T

plt.figure(figsize=(6,6))
plt.plot(np.arange(1,Nmodes+1), s_rcdt[:Nmodes]/s_rcdt[0], label='rcdt')
plt.plot(np.arange(1,Nmodes+1), s_phy[:Nmodes]/s_phy[0], label='physical')
# plt.plot(np.arange(1,Nmodes+1), s_fft[:Nmodes]/s_fft[0], label='Fourier')
plt.yscale('log')
plt.legend(['RCDT','Physical'], fontsize=18)
plt.xlabel('Modes', fontsize=18)
plt.ylabel('$\sigma / \sigma_1$', fontsize=22)
plt.xticks(ticks=[1,2,3,4,5,10,15,20], fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig('Multiphase_sv.pdf', format='pdf')

# %%

# %%
# TEST RCDT inversion only

fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(alpha_tr[k_tr,:,:].T)
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
#plt.title('Image')

plt.subplot(1,3,2)
plt.imshow(Irev[k_tr,:,:].T)
plt.colorbar(orientation='horizontal')
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
plt.clim(0,1)
#plt.title('RCDT-iRCDT{Image}')

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(alpha_tr[k_tr,:,:].T-Irev[k_tr,:,:].T)))
plt.colorbar(orientation='horizontal')
plt.clim(-2,0)
# plt.tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
# plt.title('$L_2$ Error: {:.3e}'.format(err_rcdt[k_tr]))

plt.tight_layout()
plt.savefig('Multiphase_intrinsic.pdf', format='pdf')