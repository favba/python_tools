#!/usr/bin/env python
import numpy as np
import sys
from globalfile import readglobal
import datetime
from joblib import Parallel, delayed  
import multiprocessing
import numba as nb
import os
from decomp import propdecomp

@nb.jit(cache=True,nopython=True)
def pf(t11,t11m,t22,t22m,t33,t33m,t12,t12m,t13,t13m,t23,t23m,
       d12,d13,d23,d11,d22,d33,
       index1,alpha,tm11,tm22,tm33,tm12,tm13,tm23):

  Nx, Ny = d11.shape
  hpi=2/np.pi
  
  for j in range(Ny):
    for i in range(Nx):
      b11=(2/3)*t11[i,j]-(1/3)*(t22[i,j]+t33[i,j]) - t11m[i,j]
      b22=(2/3)*t22[i,j]-(1/3)*(t11[i,j]+t33[i,j]) - t22m[i,j]
      b33=(2/3)*t33[i,j]-(1/3)*(t11[i,j]+t22[i,j]) - t33m[i,j]
      b12=t12[i,j] - t12m[i,j]
      b13=t13[i,j] - t13m[i,j]
      b23=t23[i,j] - t23m[i,j]
      e12=d12[i,j]
      e13=d13[i,j]
      e23=d23[i,j]
      e11=d11[i,j]
      e22=d22[i,j]
      e33=d33[i,j]
    
      tm11[i,j],tm22[i,j],tm33[i,j],tm12[i,j],tm13[i,j],tm23[i,j],alpha[i,j] = propdecomp(b11,b22,b33,b12,b13,b23,e11,e22,e33,e12,e13,e23)
 
      tm12[i,j]=tm12[i,j] + t12m[i,j]
      tm13[i,j]=tm13[i,j] + t13m[i,j]
      tm23[i,j]=tm23[i,j] + t23m[i,j]
      tm11[i,j]=tm11[i,j] + t11m[i,j]
      tm22[i,j]=tm22[i,j] + t22m[i,j]
      tm33[i,j]=tm33[i,j] + t33m[i,j]

      modm=np.sqrt((tm11[i,j]**2 + 2*tm12[i,j]**2 + 2*tm13[i,j]**2 + tm22[i,j]**2 + 2*tm23[i,j]**2 + tm33[i,j]**2))
      modb=np.sqrt((((2/3)*t11[i,j]-(1/3)*(t22[i,j]+t33[i,j]))**2 + 2*t12[i,j]**2 + 2*t13[i,j]**2 + ((2/3)*t22[i,j]-(1/3)*(t11[i,j]+t33[i,j]))**2 + 2*t23[i,j]**2 + ((2/3)*t33[i,j]-(1/3)*(t11[i,j]+t22[i,j]))**2))
    
      index1[i,j] = 1 - hpi*np.arccos(modm/modb)
      # index1[i,j] = modm/modb
 
  return


Fil = sys.argv[1]
N = sys.argv[2]
nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="../../../")

dtype = 'Float64'
if os.path.getsize(Fil+'P11_N'+N) == nx*ny*nz*4 : dtype = 'Float32'


d11 = np.memmap(Fil+'P11_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d12 = np.memmap(Fil+'P12_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d13 = np.memmap(Fil+'P13_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d22 = np.memmap(Fil+'P22_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d23 = np.memmap(Fil+'P23_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d33 = np.memmap(Fil+'P33_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')

t11o = np.memmap(Fil+'T11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t12o = np.memmap(Fil+'T12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t13o = np.memmap(Fil+'T13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t22o = np.memmap(Fil+'T22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t23o = np.memmap(Fil+'T23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t33o = np.memmap(Fil+'T33_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

t11m = np.memmap('./Model_II/Tm11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t12m = np.memmap('./Model_II/Tm12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t13m = np.memmap('./Model_II/Tm13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t22m = np.memmap('./Model_II/Tm22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t23m = np.memmap('./Model_II/Tm23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t33m = np.memmap('./Model_II/Tm33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

pathoutput = './Model_VI'

if not os.path.exists(pathoutput): os.makedirs(pathoutput)

index1 = np.memmap('./Model_VI/index6',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
alpha = np.memmap('./Model_VI/alpha',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm11 = np.memmap('./Model_VI/Tm11',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm12 = np.memmap('./Model_VI/Tm12',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm13 = np.memmap('./Model_VI/Tm13',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm22 = np.memmap('./Model_VI/Tm22',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm23 = np.memmap('./Model_VI/Tm23',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm33 = np.memmap('./Model_VI/Tm33',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')

a = datetime.datetime.now().replace(microsecond=0)
num_cores = multiprocessing.cpu_count()

print('Performing Calculations in {} cores'.format(num_cores))

Parallel(n_jobs=num_cores)(delayed(pf)(t11o[:,:,k],t11m[:,:,k],t22o[:,:,k],t22m[:,:,k],t33o[:,:,k],t33m[:,:,k],
    t12o[:,:,k],t12m[:,:,k],t13o[:,:,k],t13m[:,:,k],t23o[:,:,k],t23m[:,:,k],
	  d12[:,:,k],d13[:,:,k],d23[:,:,k],d11[:,:,k],d22[:,:,k],d33[:,:,k],
	  index1[:,:,k],alpha[:,:,k],tm11[:,:,k],tm22[:,:,k],tm33[:,:,k],tm12[:,:,k],tm13[:,:,k],tm23[:,:,k]) for k in range(nz))  

print('Done\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))
