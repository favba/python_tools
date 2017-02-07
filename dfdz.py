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

def part2p(field,output):
  for i in range(field.shape[0]):
    output[i,:] = fftpack.diff(field[i,:],period=pz)

  return


nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="./")
px = 2*np.pi*xDomainSize
py = 2*np.pi*yDomainSize
pz = 2*np.pi*zDomainSize

sizefile = os.path.getsize(sys.argv[1])

if sizefile == (nx+2)*ny*nz*8 :
  dtype='float64'
elif sizefile == (nx+2)*ny*nz*4 :
  dtype='float32'



field = np.memmap(sys.argv[1],shape=(nx+2,ny,nz),order='F',dtype=dtype,mode='r')[:-2,:,:]
output = np.memmap('d'+sys.argv[1]+'dz',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')

a = datetime.datetime.now().replace(microsecond=0)  
num_cores = cpu_count()

print('Performing z derivatives\n')
Parallel(n_jobs=num_cores)(delayed(part2p)(field[:,j,:],output[:,j,:]) for j in range(field.shape[1]))


print('Done\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))
