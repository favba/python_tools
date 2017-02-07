#!/usr/bin/env python

import numpy as np
import sys
from globalfile import readglobal
from numba import jit
import datetime
from joblib import Parallel, delayed  
from multiprocessing import cpu_count
import os


def calcmod(t11,t12,t13,t22,t23,t33,modt):
  calcmod2(t11,t12,t13,t22,t23,t33,modt)
  return


def tcalcmod(t11,t12,t13,t22,t23,t33,modt):
  tcalcmod2(t11,t12,t13,t22,t23,t33,modt)
  return

@jit(nogil=True)
def calcmod2(t11,t12,t13,t22,t23,t33,modt):
  modt[:,:] = np.sqrt(((2/3)*t11-(1/3)*(t22+t33))**2 + 2*t12**2 + 2*t13**2 + ((2/3)*t22-(1/3)*(t11+t33))**2 + 2*t23**2 + ((2/3)*t33-(1/3)*(t11+t22))**2)
#  modt[:,:] = modt[:,:]/modt[:,:].max()
  return

@jit(nogil=True)
def tcalcmod2(t11,t12,t13,t22,t23,t33,modt):
  modt[:,:] = np.sqrt(t11**2 + 2*t12**2 + 2*t13**2 + t22**2 + 2*t23**2 + t33**2)
#  modt[:,:] = modt[:,:]/modt[:,:].max()
  return


if len(sys.argv) == 3 :
  traceless = False
  Fil = sys.argv[1]
  N = sys.argv[2]
  nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal("../../../")

  sizefile = os.path.getsize(Fil+'T11_N'+N)
  if sizefile == nx*ny*nz*8 :
    dtype='float64'
  elif sizefile == nx*ny*nz*4 :
    dtype='float32'

  t11 = np.memmap(Fil+'T11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t12 = np.memmap(Fil+'T12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t13 = np.memmap(Fil+'T13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t22 = np.memmap(Fil+'T22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t23 = np.memmap(Fil+'T23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t33 = np.memmap(Fil+'T33_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

  modt = np.memmap(Fil+'modT_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')

elif len(sys.argv) == 1 :
  traceless = True
  nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal("../../../../")

  sizefile = os.path.getsize('Tm11')
  if sizefile == nx*ny*nz*8 :
    dtype='float64'
  elif sizefile == nx*ny*nz*4 :
    dtype='float32'


  t11 = np.memmap('Tm11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t12 = np.memmap('Tm12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t13 = np.memmap('Tm13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t22 = np.memmap('Tm22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t23 = np.memmap('Tm23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  t33 = np.memmap('Tm33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

  modt = np.memmap('modT',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')


a = datetime.datetime.now().replace(microsecond=0)
num_cores = cpu_count()

print('Performing Calculations in {} cores'.format(num_cores))

if not traceless:
  Parallel(n_jobs=num_cores,backend='threading')(delayed(calcmod2)(t11[:,:,k],t12[:,:,k],t13[:,:,k],t22[:,:,k],t23[:,:,k],t33[:,:,k],
					      modt[:,:,k]) for k in range(t11.shape[2]))
elif traceless:Parallel(n_jobs=num_cores,backend='threading')(delayed(tcalcmod2)(t11[:,:,k],t12[:,:,k],t13[:,:,k],t22[:,:,k],t23[:,:,k],t33[:,:,k],
					      modt[:,:,k]) for k in range(t11.shape[2]))

  

print('Done\n')

b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))



