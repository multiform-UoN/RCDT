# %% Imports

from time import time
import numpy as np
import matplotlib.pyplot as plt
from pytranskit.optrans.continuous.radoncdt import RadonCDT

# Database module
from ezyrb import Database

# Dimensionality reduction methods
from ezyrb import POD

# Approximation/interpolation methods
from ezyrb import RBF

# Model order reduction calss
from ezyrb import ReducedOrderModel as ROM

# FFT for paper example
from scipy.fft import fft2, ifft2


# %% Create Image Arrays

# Number of datapoints in x,y (resolution)
N = 100

# Number of time points for data
Nt = 100

# Number of POD modes to use in reconstruction
Nmodes = 20

# meshgrid from x,y arrays
x = np.linspace(0, 100, N)
y = np.linspace(0, 100, N)
x, y = np.meshgrid(x, y)

# Image array, normalised image array
ims = np.empty((Nt,N,N))
ims_norm = np.empty(ims.shape)
# FFT array for paper
ims_fft=np.empty(ims.shape)
ims_fft_imag=np.empty(ims.shape)
# Normalisation constants array
norm = np.empty(Nt)

'''
    The two parameters of a Gaussian, eta and sigma (for x and y resp.) are varied randomly.
    Note: At the moment works best for only randomly varying one at a time, and not both
    simultaneously for ROM construction. Code is set up assuming this for now.
'''
# Location of wave - random or fixed
etax = np.random.rand(Nt)*80 + 10
etay = np.random.rand(Nt)*80 + 10
#etax = np.ones(Nt)*50
#etay = np.ones(Nt)*50

# Standard Deviation - random or fixed
#sigmax = 2+np.random.rand(Nt)*10
#sigmay = 2+np.random.rand(Nt)*10
sigmax = np.ones(Nt)*5.3
sigmay = np.ones(Nt)*5.3

# Create gaussians for all times
for i in range(Nt):
    # Gaussian array
    ims[i] = np.exp(-(x-etax[i])**2 / sigmax[i]**2 - (y-etay[i])**2 / sigmay[i]**2)
    # Get norm constants, calc normalised Gaussian array
    norm[i] = np.sum(ims[i])
    ims_norm[i] = ims[i]/norm[i]
    # # FFT array for paper
    # ims_fft[i],ims_fft_imag[i] = fft2(ims_norm[i])


# %% ROM Creation

# Get image shape
dims = ims_norm.shape
# Reshape image array into 2D for use in ROM
data = np.reshape(ims_norm, (dims[0], np.prod(dims[1:])))
# Timesteps
ts = np.linspace(0,Nt,100)
# Parameters
params = np.array([etax,etay]).T

# Choose params for ROM output - depending on which variable randomised
if etax[0] == etax[1]:
    params = np.array([sigmax, sigmay]).T
elif sigmax[0] == sigmax[1]:
    params = np.array([etax, etay]).T

# ROM construction using EZyRB
db = Database(params, data)
pod = POD(rank=Nmodes) # Pre-calculated so we can get singular values from POD
rom = ROM(db, pod, linear.Linear())
# rom = ROM(db, pod, RBF())
rom.fit();

# Get mode singular values
s_phy = pod.singular_values

# # %% FFT (for paper)

# # Reshaping for ROM
# data_fft = np.reshape(ims_fft, (dims[0], np.prod(dims[1:])))

# # ROM construction
# db_fft = Database(params, data_fft)
# pod_fft = POD(rank=Nmodes)
# rom_fft = ROM(db_fft, pod_fft, RBF())
# rom_fft.fit();

# # Get mode singular values
# s_fft = pod_fft.singular_values

# %% RCDT 

# RCDT object
rcdt = RadonCDT()

# Template array for RCDT (can be an image, array of ones etc.)
templ = np.ones_like(ims[0])
# RCDT template to get RCDT space array shape
temp = rcdt.forward([0,1], templ, [0,1], templ)

# Make RCDT result array
Ihat = np.empty((Nt, np.size(temp,0), np.size(temp,1)))

# Carry out RCDT at each time
for j in range(Nt):
    Ihat[j] = rcdt.forward([0,1], templ, [0,1], ims_norm[j])

# Reshape Ihat for use in EZyRB code
dims = np.shape(Ihat)
Ihat = np.reshape(Ihat, (dims[0], np.prod(dims[1:])))

# %% RCDT - ROM Creation (using EZyRB)
    
'''
Perform POD in RCDT space using Ihat array, reshaped to use in ROM.
'''

db_rcdt = Database(params, Ihat)
rcdt_pod = POD(rank=Nmodes)
rom_rcdt = ROM(db_rcdt, rcdt_pod, RBF())
rom_rcdt.fit();

# Singular values
s_rcdt = rcdt_pod.singular_values


# %% ROM results

k = 3 # Snapshot to use for results

# Params for ROM output - depending on which variable randomised
if etax[0] == etax[1]:
    new_params = np.array([sigmax, sigmay])
elif sigmax[0] == sigmax[1]:
    new_params = np.array([etax, etay])

rom_result = np.zeros([Nt, N, N])
rcdt_rom = np.zeros((Nt,)+temp.shape)
rcdt_rom_inv = np.zeros([Nt, N, N])
err_rom = np.zeros(Nt)
err_rcdt_rom = np.zeros(Nt)

for ip in range(Nt):
    rom_result_temp = rom.predict(new_params[:,ip])
    rom_result_temp = np.reshape(rom_result_temp, (N,N))
    rom_result[ip] = rom_result_temp*norm[ip]

    # Errors
    err_rom[ip] = np.linalg.norm(rom_result[ip] - ims[ip]) / np.linalg.norm(ims[ip])
    
    # RCDT ROM
    # Predict output for snapshot selected
    rcdt_rom_temp = rom_rcdt.predict(new_params[:,ip])
    rcdt_rom[ip] = np.reshape(rcdt_rom_temp, temp.shape)
    
    # Inverse RCDT back to physical space, return to original value range
    rcdt_rom_inv[ip] = rcdt.inverse(rcdt_rom[ip], templ, [0,1])
    rcdt_rom_inv[ip] *= norm[ip]
    
    # L2 Error
    err_rcdt_rom[ip] = np.linalg.norm(rcdt_rom_inv[ip] - ims[ip]) / np.linalg.norm(ims[ip])

# %% ROM Radon-CDT predictive time interpolation

# Store predictive time interpolations: Nt-2 intermediate 'known' snapshots i.e. minusing start & end
interp_rcdt_rom = np.zeros((Nt-2,)+temp.shape)
interp_rcdt_rom_inv = np.zeros([Nt-2, N, N])
interp_err_rcdtROM = np.zeros([Nt-2])

# Loop rcdt_rom[Nt,,] for it= 0,1,2,...,Nt-1 inter-timepoint indexes to interp
for it in range(Nt-2):
    
    # Linear interpolation, using snapshots saddling desired time, in Radon-CDT space
    interp_rcdt_rom[it] = (rcdt_rom[it] + rcdt_rom[it+2])/2

    # Invert back to physical space and de-normalise !!!!NO PADDING RN!!!!
    interp_rcdt_rom_inv_temp = rcdt.inverse(interp_rcdt_rom[it],templ,[0,1])
    '''ADD PADDDING OPTIONALITY HERE'''
    interp_rcdt_rom_inv[it] = interp_rcdt_rom_inv_temp
    
    # De-normalise using normed alpha, a shifted index for same timepoint/snapshot
    interp_rcdt_rom_inv[it] *= norm[it+1]
    # Calculate L2 rel. err.
    interp_err_rcdtROM[it] = np.linalg.norm(interp_rcdt_rom_inv[it,::-1,:] - ims[it+1,::-1,:]) / np.linalg.norm(ims[it+1,::-1,:])

# get index of min. rel. L2 err. w.r.t time index/snapshot
interp_min_err_index = np.argmin(interp_err_rcdtROM)
# figure and axis for min. err. predictive interpolated plot snapshot & err. over time
figI, axsI = plt.subplots(1,2,figsize=(15,9))

# Construct figures of min. rel. err. interpolated snapshot & rel. err. over time
temp_fig = axsI[0].imshow(interp_rcdt_rom_inv[interp_min_err_index,::-1,:])
figI.colorbar(temp_fig,ax=axsI[0], orientation="horizontal", ticks=[0,0.25,0.5,0.75,1])
temp_fig.set_clim(0,1)

temp_fig = axsI[1].plot(np.arange(1,99),interp_err_rcdtROM[:])

for i in range(1):
    axsI[i].tick_params(left = False, right = False , labelleft = False ,
                             labelbottom = False, bottom = False)

plt.tight_layout()
print('Mean predictive interp. RCDT-POD error: {:.3e}'.format(np.mean(interp_err_rcdtROM)))
print('Snapshot with min. error displayed. Error over time, indexed by snapshot, graphed')
# %% Plots for both ROMS

fig, axs = plt.subplots(2,3,figsize=(15,10))

temp_fig = axs[0,0].imshow(ims[k])
fig.colorbar(temp_fig, ax=axs[0,0],orientation="horizontal")
temp_fig.set_clim(0,1)

temp_fig = axs[0,1].imshow(rom_result[k])
fig.colorbar(temp_fig, ax=axs[0,1],orientation="horizontal")
temp_fig.set_clim(0,1)

temp_fig = axs[0,2].imshow(rom_result[k]-ims[k])
fig.colorbar(temp_fig, ax=axs[0,2],orientation="horizontal")
temp_fig.set_clim(-0.35,0.35)
    
#axs[0,0].set_title("Full order")
#axs[0,1].set_title("Physical ROM (POD+RBF)")
#axs[0,2].set_title("$L_2$ Error: {:.3e}".format(err_rom[k]))

axs[1,0].axis('off')
temp_fig = axs[1,1].imshow(rcdt_rom_inv[k])
fig.colorbar(temp_fig, ax=axs[1,1],orientation="horizontal")
temp_fig.set_clim(0,1)

temp_fig = axs[1,2].imshow(rcdt_rom_inv[k] - ims[k])
fig.colorbar(temp_fig, ax=axs[1,2],orientation="horizontal")
temp_fig.set_clim(-0.35,0.35)

#axs[1,0].set_title("Full order")
#axs[1,1].set_title("RCDT ROM (POD - {:d} modes)".format(Nmodes))
#axs[1,2].set_title("$L_2$ Error: {:.3e}".format(err_rcdt_rom[k]))

for i in range(2):
    for j in range(3):
        axs[i,j].tick_params(left = False, right = False , labelleft = False ,
                             labelbottom = False, bottom = False)
plt.tight_layout()

# %% POD mode comparison

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
plt.savefig('Gaussian_sv.pdf', format='pdf')
