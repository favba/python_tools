#!/usr/bin/env python
import numpy as np
import sys
from globalfile import readglobal
import datetime
from joblib import Parallel, delayed  
import multiprocessing
import numba
from tempfile import TemporaryFile
import os

@numba.jit(nopython=True,cache=True)
def pfunc(t11,t22,t33,t12,t13,t23,
      m1t11,m1t22,m1t33,m1t12,m1t13,m1t23,
      m3t11,m3t22,m3t33,m3t12,m3t13,m3t23,
      tm11,tm22,tm33,tm12,tm13,tm23,index):
  
  Nx, Ny = t11.shape
  hpi=2/np.pi 
  for j in range(Ny):
    for i in range(Nx):
      t11r = (2/3)*t11[i,j]-(1/3)*(t22[i,j]+t33[i,j])
      t22r = (2/3)*t22[i,j]-(1/3)*(t11[i,j]+t33[i,j])
      t33r = (2/3)*t33[i,j]-(1/3)*(t11[i,j]+t22[i,j])
      
      tm11[i,j] = m1t11[i,j] + m3t11[i,j]
      tm12[i,j] = m1t12[i,j] + m3t12[i,j]
      tm13[i,j] = m1t13[i,j] + m3t13[i,j]
      tm22[i,j] = m1t22[i,j] + m3t22[i,j]
      tm23[i,j] = m1t23[i,j] + m3t23[i,j]
      tm33[i,j] = m1t33[i,j] + m3t33[i,j]

      modm=np.sqrt((tm11[i,j]**2 + 2*tm12[i,j]**2 + 2*tm13[i,j]**2 + tm22[i,j]**2 + 2*tm23[i,j]**2 + tm33[i,j]**2))
      modb=np.sqrt((t11r**2 + 2*t12[i,j]**2 + 2*t13[i,j]**2 + t22r**2 + 2*t23[i,j]**2 + t33r**2))
 
      index[i,j] = 1 - hpi*np.arccos(modm/modb)
      # index[i,j] = modm/modb
  
  
  return

def f(t11,t22,t33,t12,t13,t23,
      m1t11,m1t22,m1t33,m1t12,m1t13,m1t23,
      m3t11,m3t22,m3t33,m3t12,m3t13,m3t23,
      tm11,tm22,tm33,tm12,tm13,tm23,index):
  
  pfunc(t11[:,:],t22[:,:],t33[:,:],t12[:,:],t13[:,:],t23[:,:],
      m1t11[:,:],m1t22[:,:],m1t33[:,:],m1t12[:,:],m1t13[:,:],m1t23[:,:],
      m3t11[:,:],m3t22[:,:],m3t33[:,:],m3t12[:,:],m3t13[:,:],m3t23[:,:],
      tm11[:,:],tm22[:,:],tm33[:,:],tm12[:,:],tm13[:,:],tm23[:,:],index[:,:])
  
  return


Fil = sys.argv[1]
N = sys.argv[2]
nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="../../../")

dtype= 'float64'
if os.path.getsize(Fil+'T11_N'+N) == nx*nz*ny*4 : dtype = 'float32'

t11 = np.memmap(Fil+'T11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t12 = np.memmap(Fil+'T12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t13 = np.memmap(Fil+'T13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t22 = np.memmap(Fil+'T22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t23 = np.memmap(Fil+'T23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t33 = np.memmap(Fil+'T33_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')


m1t11 = np.memmap('./Model_I/Tm11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m1t12 = np.memmap('./Model_I/Tm12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m1t13 = np.memmap('./Model_I/Tm13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m1t22 = np.memmap('./Model_I/Tm22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m1t23 = np.memmap('./Model_I/Tm23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m1t33 = np.memmap('./Model_I/Tm33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

m3t11 = np.memmap('./Model_III/Tm11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m3t12 = np.memmap('./Model_III/Tm12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m3t13 = np.memmap('./Model_III/Tm13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m3t22 = np.memmap('./Model_III/Tm22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m3t23 = np.memmap('./Model_III/Tm23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
m3t33 = np.memmap('./Model_III/Tm33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

if not os.path.exists('./Model_V'): os.mkdir('./Model_V')

tm11 = np.memmap('./Model_V/Tm11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm12 = np.memmap('./Model_V/Tm12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm13 = np.memmap('./Model_V/Tm13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm22 = np.memmap('./Model_V/Tm22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm23 = np.memmap('./Model_V/Tm23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm33 = np.memmap('./Model_V/Tm33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
index = np.memmap('./Model_V/index',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')


a = datetime.datetime.now().replace(microsecond=0)
num_cores = multiprocessing.cpu_count()

print('Performing Calculations in {} cores'.format(num_cores))


Parallel(n_jobs=num_cores)(delayed(pfunc)(t11[:,:,k],t22[:,:,k],t33[:,:,k],t12[:,:,k],t13[:,:,k],t23[:,:,k],
       m1t11[:,:,k],m1t22[:,:,k],m1t33[:,:,k],m1t12[:,:,k],m1t13[:,:,k],m1t23[:,:,k],
       m3t11[:,:,k],m3t22[:,:,k],m3t33[:,:,k],m3t12[:,:,k],m3t13[:,:,k],m3t23[:,:,k],
       tm11[:,:,k],tm22[:,:,k],tm33[:,:,k],tm12[:,:,k],tm13[:,:,k],tm23[:,:,k],index[:,:,k]) for k in range(t11.shape[2]))  

print('Done\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))
