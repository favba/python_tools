#!/usr/bin/env python

import numpy as np
import sys
from globalfile import readglobal
from numba import jit
from scipy import fftpack
import datetime
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import os

def part1p(field,px,py,g1,g2):
  for j in range(field.shape[1]):
    g1[:,j] = fftpack.diff(field[:,j],period=px)
  for i in range(field.shape[0]):
    g2[i,:] = fftpack.diff(field[i,:],period=py)
  return

def part2p(field,pz,g3):
  for i in range(field.shape[0]):
    g3[i,:] = fftpack.diff(field[i,:],period=pz)

  return

sizefile = os.path.getsize(sys.argv[1])

nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="../../../")
px = 2*np.pi*xDomainSize
py = 2*np.pi*yDomainSize
pz = 2*np.pi*zDomainSize

if sizefile == nx*ny*nz*8 :
  dtype='float64'
elif sizefile == nx*ny*nz*4 :
  dtype='float32'

a = datetime.datetime.now().replace(microsecond=0)

g1 = np.memmap('d'+sys.argv[1]+'dx',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
g2 = np.memmap('d'+sys.argv[1]+'dy',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
g3 = np.memmap('d'+sys.argv[1]+'dz',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')

field = np.memmap(sys.argv[1],shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

a = datetime.datetime.now().replace(microsecond=0)  
num_cores = cpu_count()
print('Performing x and y derivatives\n')
Parallel(n_jobs=num_cores)(delayed(part1p)(field[:,:,k],px,py,g1[:,:,k],g2[:,:,k]) for k in range(field.shape[2]))

print('Performing z derivatives\n')
Parallel(n_jobs=num_cores)(delayed(part2p)(field[:,j,:],pz,g3[:,j,:]) for j in range(field.shape[1]))


print('Done\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))




