#!/usr/bin/env python
import numpy as np
import sys
from globalfile import readglobal
import datetime
from joblib import Parallel, delayed  
import multiprocessing
import numba
import os
from decomp import inphdecomp

def f(t11,t22,t33,t12,t13,t23,
       d12,d13,d23,d11,d22,d33,
       index2,alpha0,alpha1,alpha2,hpi,
       tm11,tm22,tm33,tm12,tm13,tm23):

  pf(t11,t22,t33,t12,t13,t23,
       d12,d13,d23,d11,d22,d33,
       index2,alpha0,alpha1,alpha2,hpi,
       tm11,tm22,tm33,tm12,tm13,tm23)

@numba.jit(cache=True,nopython=True)
def pf(t11,t22,t33,t12,t13,t23,
       d12,d13,d23,d11,d22,
       index2,alpha0,alpha1,alpha2,hpi,
       tm11,tm22,tm33,tm12,tm13,tm23):

  for j in range(d11.shape[1]):
    for i in range(d11.shape[0]):
      b11=(2/3)*t11[i,j]-(1/3)*(t22[i,j]+t33[i,j])
      b22=(2/3)*t22[i,j]-(1/3)*(t11[i,j]+t33[i,j])
      b33=(2/3)*t33[i,j]-(1/3)*(t11[i,j]+t22[i,j])
      b12=t12[i,j]
      b13=t13[i,j]
      b23=t23[i,j]
      e12=d12[i,j]
      e13=d13[i,j]
      e23=d23[i,j]
      e11=d11[i,j]
      e22=d22[i,j]
      e33=-e11-e22

      tm11[i,j],tm22[i,j],tm33[i,j],tm12[i,j],tm13[i,j],tm23[i,j],alpha0[i,j],alpha1[i,j],alpha2[i,j] = inphdecomp(b11,b22,b33,b12,b13,b23,e11,e22,e33,e12,e13,e23)

      modm=np.sqrt((tm11[i,j]**2 + 2*tm12[i,j]**2 + 2*tm13[i,j]**2 + tm22[i,j]**2 + 2*tm23[i,j]**2 + tm33[i,j]**2))
      modb=np.sqrt((b11**2 + 2*b12**2 + 2*b13**2 + b22**2 + 2*b23**2 + b33**2))
    
      index2[i,j] = 1 - hpi*np.arccos(modm/modb)
      # index2[i,j] = modm/modb
 
  return

Fil = sys.argv[1]
N = sys.argv[2]

try:
  nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="./")
except:
  nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="../../../")

dtype = 'float64'
if os.path.getsize(Fil+'D11_N'+N) == nx*ny*nz*4 : dtype = 'float32'

d11 = np.memmap(Fil+'D11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
d12 = np.memmap(Fil+'D12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
d13 = np.memmap(Fil+'D13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
d22 = np.memmap(Fil+'D22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
d23 = np.memmap(Fil+'D23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

t11 = np.memmap(Fil+'T11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t12 = np.memmap(Fil+'T12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t13 = np.memmap(Fil+'T13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t22 = np.memmap(Fil+'T22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t23 = np.memmap(Fil+'T23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t33 = np.memmap(Fil+'T33_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

if not os.path.exists('./Model_II'): os.mkdir('./Model_II')

index2 = np.memmap('./Model_II/index2',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
alpha0 = np.memmap('./Model_II/alpha0',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
alpha1 = np.memmap('./Model_II/alpha1',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
alpha2 = np.memmap('./Model_II/alpha2',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')

tm11 = np.memmap('./Model_II/Tm11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm12 = np.memmap('./Model_II/Tm12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm13 = np.memmap('./Model_II/Tm13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm22 = np.memmap('./Model_II/Tm22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm23 = np.memmap('./Model_II/Tm23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
tm33 = np.memmap('./Model_II/Tm33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')


a = datetime.datetime.now().replace(microsecond=0)

num_cores = multiprocessing.cpu_count()

print('Performing Calculations in {} cores'.format(num_cores))

hpi=2/np.pi

Parallel(n_jobs=num_cores)(delayed(pf)(t11[:,:,k],t22[:,:,k],t33[:,:,k],t12[:,:,k],t13[:,:,k],t23[:,:,k],
                         d12[:,:,k],d13[:,:,k],d23[:,:,k],d11[:,:,k],d22[:,:,k],
                        index2[:,:,k],alpha0[:,:,k],alpha1[:,:,k],alpha2[:,:,k],hpi,
                    tm11[:,:,k],tm22[:,:,k],tm33[:,:,k],tm12[:,:,k],tm13[:,:,k],tm23[:,:,k]) for k in range(d11.shape[2]))  


print('Done\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))
