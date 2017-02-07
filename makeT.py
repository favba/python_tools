#!/usr/bin/env python

import numpy as np
import sys
from globalfile import readglobal
import numba
import datetime
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import os

@numba.jit(cache=True,nopython=True)
def compiledop(field1,field2,field3,result):
  result[:,:] = field1[:,:] - field2[:,:]*field3[:,:]
  return

def paralleloperation(field1,field2,field3,result):
  compiledop(field1,field2,field3,result)
  return

Fil = sys.argv[1]
N = sys.argv[2]
path = "../../../"

nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path=path)

sizefile = os.path.getsize(Fil+'u1_N_'+N)

if sizefile == nx*ny*nz*8 :
  dtype='float64'
elif sizefile == nx*ny*nz*4 :
  dtype='float32'

u1 = np.memmap(Fil+'u1_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
u2 = np.memmap(Fil+'u2_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
u3 = np.memmap(Fil+'u3_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

u1u1 = np.memmap(Fil+'u1u1_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
u1u2 = np.memmap(Fil+'u1u2_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
u1u3 = np.memmap(Fil+'u1u3_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

u2u2 = np.memmap(Fil+'u2u2_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
u2u3 = np.memmap(Fil+'u2u3_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

u3u3 = np.memmap(Fil+'u3u3_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

t11 = np.memmap(Fil+'T11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
t12 = np.memmap(Fil+'T12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
t13 = np.memmap(Fil+'T13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
t22 = np.memmap(Fil+'T22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
t23 = np.memmap(Fil+'T23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
t33 = np.memmap(Fil+'T33_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')

a = datetime.datetime.now().replace(microsecond=0)

num_cores = cpu_count()

print('Performing Calculations in {} cores'.format(num_cores))
print('\nCalculating T11')
Parallel(n_jobs=num_cores)(delayed(compiledop)(u1u1[:,:,k],u1[:,:,k],u1[:,:,k],t11[:,:,k]) for k in range(t11.shape[2]))

print('\nCalculating T12')
Parallel(n_jobs=num_cores)(delayed(compiledop)(u1u2[:,:,k],u1[:,:,k],u2[:,:,k],t12[:,:,k]) for k in range(t11.shape[2]))

print('\nCalculating T13')
Parallel(n_jobs=num_cores)(delayed(compiledop)(u1u3[:,:,k],u1[:,:,k],u3[:,:,k],t13[:,:,k]) for k in range(t11.shape[2]))

print('\nCalculating T22')
Parallel(n_jobs=num_cores)(delayed(compiledop)(u2u2[:,:,k],u2[:,:,k],u2[:,:,k],t22[:,:,k]) for k in range(t11.shape[2]))

print('\nCalculating T23')
Parallel(n_jobs=num_cores)(delayed(compiledop)(u2u3[:,:,k],u2[:,:,k],u3[:,:,k],t23[:,:,k]) for k in range(t11.shape[2]))

print('\nCalculating T33')
Parallel(n_jobs=num_cores)(delayed(compiledop)(u3u3[:,:,k],u3[:,:,k],u3[:,:,k],t33[:,:,k]) for k in range(t11.shape[2]))

print('\nDone\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))
