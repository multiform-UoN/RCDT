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
# Parameters and data
datapath='../Data/airfoil/'

# POD and interpolation parameters
Nmodes = 20 # Number of modes to use for ROMs
desired_length = 1.2 # Desired length of training set (automatically reduced accordingly)

# fixed time or mach number
fixedTime = True # interpolate on Mu, fix time
fixedTimeIndex = 299 # index of time to use for fixed time
fixedMuIndex = 0 # reference Mu index for variable time

# LOAD datasets
mulist = np.load(datapath+'mu.npy') # Mu values (Mach)
timelist = np.load(datapath+'times.npy') # Time values
timelist = timelist[fixedMuIndex,:]/timelist[fixedMuIndex,0] # Normalise time values and consider only the first (times are scaled by Mu)

# Initialize an empty list to accumulate the arrays
u_orig = []

# Load the array
cropped = False
for i in range(len(mulist)):
    # Load the array and append it to the list
    u_orig.append(np.load(datapath + 'uniformVelocity_Mu['
                         + str(mulist[i]) + ']'
                         + ('cropped' if cropped else '')
                         + '.npy'))

# Convert the list of arrays to a NumPy array
u_orig = np.array(u_orig)
        
# No. snapshots
N = len(timelist)
if fixedTime:
    N = len(mulist)

# parameter space
ind = timelist
if fixedTime:
    ind = mulist
        



# %%
# Reduce data

# thresholding
thresholding = False

# Padding - add padding to image arrays before RCDT
pad = False
pad_val = (0,0) # (x,y)
pad_width = (30,30) # (x,y)
pad_type = 'constant'

# Setting training data (a reduced version of the full dataset)
skip = math.ceil(len(ind)/desired_length)
ind_tr = ind[::skip]
Ntr = len(ind_tr)
if fixedTime:
    u_tr = np.squeeze(u_orig[::skip,fixedTimeIndex,:,:])
    u = np.squeeze(u_orig[:,fixedTimeIndex,:,:])
else:
    u_tr = np.squeeze(u_orig[fixedMuIndex,::skip,:,:])
    u = np.squeeze(u_orig[fixedMuIndex,:,:,:]) 

k_tr = math.ceil(len(ind_tr)/2) # reference index to use to check training set
k = k_tr*skip - math.ceil(skip/2) # reference index to use to predict
k0 = k_tr-1  # snapshots either side of target to use for interp
k1 = k_tr # snapshots either side of target to use for interp
k0c = (ind[k] - ind_tr[k0])/(ind_tr[k1] - ind_tr[k0]) # interpolation coefficient

# No. of uniform grid points for griddata interp
Ny = int(u.shape[1])
Nx = int(u.shape[2])


# %%
# Pre-processing of inputs

# Invert velocity field based on boundary values
preprocess = True
if preprocess:
    # Compute the maximum along the third and fourth dimensions for each (i, j) pair
    # ref_values = u.max(axis=(1, 2))
    ref_values = u[:,0,0]
    # ref_values = np.max(u,0)

    # Normalize for every (i, j) pair
    u = -u/ref_values[:,np.newaxis,np.newaxis] + 1
    # u = -u/ref_values + 1

    # Compute the maximum along the third and fourth dimensions for each (i, j) pair
    # ref_values = u.max(axis=(1, 2))
    ref_values = u_tr[:,0,0]
    # ref_values = np.max(u_tr,0)

    # Normalize for every (i, j) pair
    u_tr = -u_tr/ref_values[:,np.newaxis,np.newaxis] + 1
    # u_tr = -u_tr/ref_values + 1

fig = plt.figure(figsize=(15,5))

# Show post-processed image data
i=0
for j in u_tr:
    plt.subplot(1,len(ind_tr+1),i+1)
    plt.imshow(j)
    plt.colorbar(orientation='horizontal')
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    i += 1


# vectors to pass to Ezyrb Databased
indtr = np.array(list(ind_tr)).reshape(-1,1)
snaps = np.array([u_tr[i,:,:].reshape(Nx*Ny) for i in range(Ntr)])

# %%
# Physical ROM using EZyRB

tic = time()

# Physical ROM construction    
db = Database(indtr,snaps)
phys_pod = POD(rank = Nmodes)
rom = ROM(db, phys_pod, linear.Linear())
rom.fit();

# for i in range(np.size(Nmodes)):
#     # phys_pod = POD(rank = Nmodes[i])
#     phys_podI[i] = POD(rank = Nmodes[i])
#     # rom = ROM(db, phys_pod, linear.Linear())
#     phys_romI[i] = ROM(db,phys_podI[i],linear.Linear())
#     # rom.fit();
#     phys_romI[i].fit();

# POD singular values
s_phys = phys_pod.singular_values
toc = time()
print("Physical ROM time elapsed: {:.2f} minutes".format((toc-tic)/60))


# %%
# TEST Physical ROM

# # Manual interpolation
# p = (alpha_tr[k0]*k0c + alpha_tr[k1]*(1-k0c)).T

# Interpolation using EZYRB
p = rom.predict(ind[k]).reshape(Ny,Nx).T

err_interp = np.linalg.norm(u[k,:,:]-p.T)/np.linalg.norm(u[k])


fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(u[k0*skip,:,:])
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
refmax = u[k,:,:].max()
refmin = u[k,:,:].min()
plt.clim(refmin,refmax)

plt.subplot(1,3,2)
plt.imshow(u[k,:,:])
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.clim(refmin,refmax)

plt.subplot(1,3,3)
plt.imshow(u[k1*skip,:,:])
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
plt.clim(refmin,refmax)

plt.tight_layout()

plt.savefig('Airfoil_snapshots.pdf', format='pdf')

   
fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(u[k,:,:])
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
refmax = u[k,:,:].max()
refmin = u[k,:,:].min()
plt.clim(refmin,refmax)

plt.subplot(1,3,2)
plt.imshow(p.T)
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.clim(refmin,refmax)

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(u[k,:,:]-p.T)+1e-10))
plt.colorbar(orientation='horizontal')
# plt.title('$L_2$ Error: {:.3e}'.format(np.linalg.norm((u[kRef,fixedTime,:,:]-p.T),2)/
                            # np.linalg.norm(u[kRef,fixedTime,:,:],2)))
plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
plt.clim(-3,1)

plt.tight_layout()
plt.savefig('Airfoil_phys.pdf', format='pdf')


# %%
# RCDT  data

upos = u_tr*(u_tr>0)
uneg = u_tr*(u_tr<0)*(-1)
# get normalising constants
norm_upos = np.sum(upos, (1,2))
norm_uneg = np.sum(uneg, (1,2))

# normalize velocity
for i in range(Ntr):
    upos[i] = upos[i]/norm_upos[i]
    # uneg[i] = uneg[i]/norm_uneg[i]

# # subtract max value
# maxu = np.max(upos, 0)
# upos = maxu - upos


rcdt = RadonCDT(theta=np.linspace(0,180,360)) # create RCDT object

if pad:
    temp = np.ones_like(alpha_pad[0]) # reference signal
    Irev_temp = np.zeros(alpha_pad.shape[0])
else:
    temp = np.ones_like(u_tr[0].T) # reference signal
    Irev_temp = np.zeros(u_tr.shape[0])

tic = time()

# Get shape for RCDT array
rcdt_shape = (Ntr,) + np.shape(rcdt.forward([0,1], temp, [0,1], temp))

# Arrays for RCDT calculations
Ihat = np.zeros(rcdt_shape)
Ihatneg = np.zeros(rcdt_shape)
Irev = np.zeros_like(u_tr)
err_rcdt = np.zeros(Ntr)

# Perform RCDT, iRCDT and de-normalisation
for i in range(Ntr):
    if pad:
        Ihat[i] = rcdt.forward([0,1], temp, [0,1], alpha_pad[i])

    Ihat[i] = rcdt.forward([0,1], temp, [0,1], upos[i].T)
    Ihatneg[i] = rcdt.forward([0,1], temp, [0,1], uneg[i].T)
        
    Irev_temp = norm_upos[i]*rcdt.inverse(Ihat[i], temp, [0,1]).T - norm_uneg[i]*rcdt.inverse(Ihatneg[i], temp, [0,1]).T
    # Irev_temp = norm_upos[i]*(maxu - rcdt.inverse(Ihat[i], temp, [0,1]).T)
    
    if pad:
        Irev[i] = Irev_temp[pad_width[0]:-pad_width[0], pad_width[1]:-pad_width[1]]
    else:
        Irev[i] = Irev_temp
    
    #L2-norm error
    err_rcdt[i] = np.linalg.norm(Irev[i] - u_tr[i],2) / np.linalg.norm(u_tr[i],2)
    
    
toc = time()
print("RCDT-iRCDT time elapsed: {:.2f} minutes".format((toc-tic)/60))

# %% RCDT POD

tic = time()
# Construct database object, in RCDT space, for ROM(POD) use
db = Database(indtr, np.reshape(np.stack([Ihat,Ihatneg],-1), (Ntr, -1)))
# db = Database(indtr, np.reshape(Ihat,(Ntr, -1)))
# Construct POD object; default, singular val. decomp (SVD); for Nmodes rank
rcdt_pod = POD(rank = Nmodes)
# Perform reduce order modelling (ROM) i.e. POD, linear interpolation
rom_rcdt = ROM(db, rcdt_pod, linear.Linear())
rom_rcdt.fit();
# Retrieve singular vals. of decomp
s_rcdt = rcdt_pod.singular_values

toc = time()
print("RCDT ROM time elapsed: {:.2f} minutes".format((toc-tic)/60))

# %%
# TEST RCDT inversion only

fig = plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(u_tr[k_tr,:,:])
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
refmax = u_tr[k_tr,:,:].max()
refmin = u_tr[k_tr,:,:].min()
plt.clim(refmin,refmax)

plt.subplot(1,3,2)
plt.imshow(Irev[k_tr,:,:])
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
plt.clim(refmin,refmax)

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(u_tr[k_tr,:,:]-Irev[k_tr,:,:])))
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
# plt.title('$L_2$ Error: {:.3e}'.format(err_rcdt[k_trRef]))
plt.clim(-3,1)

plt.tight_layout()

plt.savefig('Airfoil_intrinsic.pdf', format='pdf')


# %%
# TEST RCDT interpolation

# Manual interpolation
# Ihat_interp = (Ihat[k0]*k0c + Ihat[k1]*(1-k0c))

# Interpolation using EZYRB (sign split)
Ihat_interp = rom_rcdt.predict(ind[k]).reshape(rcdt_shape[1:]+ (2,))

Irev_interp = rcdt.inverse(Ihat_interp[...,0], temp, [0,1])
Irev_interp_neg = rcdt.inverse(Ihat_interp[...,1], temp, [0,1])
Irev_interp *= (norm_upos[k0]*k0c + norm_upos[k1]*(1-k0c))
Irev_interp_neg *= (norm_uneg[k0]*k0c + norm_uneg[k1]*(1-k0c))

# # # Interpolation using EZYRB
# Ihat_interp = rom_rcdt.predict(ind[k]).reshape(rcdt_shape[1:])

# Irev_interp = rcdt.inverse(Ihat_interp[...], temp, [0,1])
# Irev_interp *= (norm_upos[k0]*k0c + norm_upos[k1]*(1-k0c))

if pad:
    Irev_interp = Irev_interp[pad_width[0]:-pad_width[0], pad_width[1]:-pad_width[1]]

p = Irev_interp - Irev_interp_neg
# p = maxu - Irev_interp

# L2-norm error
err_interp_rcdt = np.linalg.norm(Irev_interp - u[k].T)/np.linalg.norm(u[k])

plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.imshow(u[k,:,:])
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
refmax = u[k,:,:].max()
refmin = u[k,:,:].min()
plt.clim(refmin,refmax)

plt.subplot(1,3,2)
plt.imshow(p.T)
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.clim(refmin,refmax)

plt.subplot(1,3,3)
plt.imshow(np.log10(abs(u[k,:,:]-p.T)))
plt.colorbar(orientation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
# plt.title('$L_2$ Error: {:.3e}'.format(np.linalg.norm((u[kRef,fixedTime,:,:]-p.T),2)/
#                                     np.linalg.norm(u[kRef,fixedTime,:,:],2)))
plt.clim(-3,1)
    
plt.tight_layout()

plt.savefig('Airfoil_rcdt.pdf', format='pdf')

# %% 
# POD Modes decay

plt.figure(figsize=(6,6))
plt.plot(np.arange(1,s_rcdt.size+1), s_rcdt/s_rcdt[0], label='rcdt')
plt.plot(np.arange(1,s_phys.size+1), s_phys/s_phys[0], label='physical')
plt.yscale('log')
plt.legend(['RCDT','Physical'], fontsize=18)
plt.xlabel('Mode', fontsize=18)
plt.ylabel('$\sigma / \sigma_1$', fontsize=22)
plt.xticks(ticks=[1,2,3,4,5,10,15,20], fontsize=14)
plt.yticks(fontsize=14)


plt.tight_layout()
plt.savefig('Airfoil_sv'+str(fixedTime)+'.pdf', format='pdf')

