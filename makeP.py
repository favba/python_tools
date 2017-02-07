#!/usr/bin/env python

import numpy as np
import sys
from globalfile import readglobal
import numba as nb
import datetime
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import os

@nb.jit(cache=True,nopython=True)
def cf1(d11,d12,d13,d22,d23,d33,w12,w13,w23,p11,p12,p13,p22,p23,p33):
  p11[:,:]= -2*(d12*w12 + d13*w13)
  p22[:,:]= 2*(d12*w12 - d23*w23)
  p33[:,:]= 2*(d13*w13 + d23*w23)
  p12[:,:] = w12*(d11-d22) -d13*w23 - d23*w13
  p13[:,:] = w13*(d11-d33) + d12*w23 - d23*w12
  p23[:,:] = w23*(d22-d33) + d12*w13 + d13*w12
  return

def pf1(d11,d12,d13,d22,d23,d33,w12,w13,w23,p11,p12,p13,p22,p23,p33):
  cf1(d11,d12,d13,d22,d23,d33,w12,w13,w23,p11,p12,p13,p22,p23,p33)
  return


if len(sys.argv) == 3:
  Fil = sys.argv[1]
  N = sys.argv[2]
  path = "../../../"

  nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path=path)

  sizefile = os.path.getsize(Fil+'D11_N'+N)
  if sizefile == nx*ny*nz*8 :
    dtype='float64'
  elif sizefile == nx*ny*nz*4 :
    dtype='float32'

  d11 = np.memmap(Fil+'D11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d12 = np.memmap(Fil+'D12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d13 = np.memmap(Fil+'D13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d22 = np.memmap(Fil+'D22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d23 = np.memmap(Fil+'D23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d33 = np.memmap(Fil+'D33_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

  w12 = np.memmap(Fil+'W12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  w13 = np.memmap(Fil+'W13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  w23 = np.memmap(Fil+'W23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')


  p11 = np.memmap(Fil+'P11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p12 = np.memmap(Fil+'P12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p13 = np.memmap(Fil+'P13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p22 = np.memmap(Fil+'P22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p23 = np.memmap(Fil+'P23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p33 = np.memmap(Fil+'P33_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')  

elif len(sys.argv) == 1:
  nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="./")

  sizefile = os.path.getsize('D11')
  if sizefile == nx*ny*nz*8 :
    dtype='float64'
  elif sizefile == nx*ny*nz*4 :
    dtype='float32'

  d11 = np.memmap('D11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d12 = np.memmap('D12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d13 = np.memmap('D13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d22 = np.memmap('D22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d23 = np.memmap('D23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  d33 = np.memmap('D33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

  w12 = np.memmap('W12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  w13 = np.memmap('W13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  w23 = np.memmap('W23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')


  p11 = np.memmap('P11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p12 = np.memmap('P12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p13 = np.memmap('P13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p22 = np.memmap('P22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p23 = np.memmap('P23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  p33 = np.memmap('P33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')  

a = datetime.datetime.now().replace(microsecond=0)

num_cores = cpu_count()

print('Performing Calculations in {} cores'.format(num_cores))
print('\nCalculating ...')
Parallel(n_jobs=num_cores)(delayed(cf1)(d11[:,:,k],d12[:,:,k],d13[:,:,k],d22[:,:,k],d23[:,:,k],d33[:,:,k],w12[:,:,k],w13[:,:,k],w23[:,:,k],p11[:,:,k],p12[:,:,k],p13[:,:,k],p22[:,:,k],p23[:,:,k],p33[:,:,k]) for k in range(d11.shape[2]))


print('\nDone\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))
