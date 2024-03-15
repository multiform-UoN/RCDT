#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import pymplu
plt.rcParams['text.usetex'] = True


# In[95]:


def gaussian_function(sigma,mu,x):
    I = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((x-mu)**2)/(2*sigma**2))
    return I
N=250
x0 = np.linspace(0, 1, N)
I0= np.ones(x0.size)
x=np.linspace(0, 2, N)
mu_list = np.linspace(0.25,1.75,21)
sigma=0.1
gaussian_list = []
for i in mu_list:
    gaussian_list.append(gaussian_function(sigma,i,x))
gaussian_list_plot = gaussian_list[0:None:5]
mu_list_plot = mu_list[0:None:5]
i=0
for f, mu in zip(gaussian_list_plot, mu_list_plot):
    plt.plot(x,f,label=r'$\mu =$ '+str(mu),color=pymplu.colors[i],marker=pymplu.markers[i],markevery=10)
    i=i+1

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, ncol=5)
pymplu.set_default_font()
plt.ylabel(r'$f(x;\mu)$')
plt.xlabel(r'$x$')
plt.savefig("snaps.pdf",bbox_inches='tight')


# In[94]:


import sys
sys.path.append('../')

from pytranskit.optrans.continuous.cdt import CDT

# Create a CDT object
cdt = CDT()

I_hat_list = []

for i in gaussian_list:
    I1_hat, I1_hat_old, xtilde = cdt.forward(x0, I0, x, i)
    I_hat_list.append(I1_hat)

I_hat_list_plot = I_hat_list[0:None:5]
i = 0
for f, mu in zip(I_hat_list_plot, mu_list_plot):
    plt.plot(xtilde,f,label=r'$\mu =$ '+str(mu),color=pymplu.colors[i],marker=pymplu.markers[i],markevery=10)
    i=i+1

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, ncol=5)
pymplu.set_default_font()
plt.ylabel(r'$\hat{f}(\hat{x};\mu)$')
plt.xlabel(r'$\hat{x}$')
plt.savefig("snaps_trans.pdf",bbox_inches='tight')


# In[34]:


I_hat_list_matrix = (np.vstack(I_hat_list)).T
S_matrix = (np.vstack(gaussian_list)).T

U1, S1, Vh1 = np.linalg.svd(S_matrix, full_matrices=False)
U2, S2, Vh2 = np.linalg.svd(I_hat_list_matrix, full_matrices=False)

plt.plot(S1,label="Physical Space")
plt.plot(S2,label="CDT space")
plt.ylabel(r'$\sigma$')
plt.xlabel(r'$N$')
plt.legend()
plt.show()

