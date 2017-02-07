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

def part1p(d11,d12,d13,d22,d23,d33,u1,u2,u3,px,py,w12,w13,w23):
  for j in range(u1.shape[1]):
    d11[:,j] = fftpack.diff(u1[:,j],period=px)
    d12[:,j] = fftpack.diff(u2[:,j],period=px)
    w12[:,j] = d12[:,j]
    d13[:,j] = fftpack.diff(u3[:,j],period=px)
    w13[:,j] = d13[:,j]
  for i in range(u1.shape[0]):
    aux = fftpack.diff(u1[i,:],period=py)
    d12[i,:] = (d12[i,:]+aux)/2
    w12[i,:] = (w12[i,:]-aux)/2
    d22[i,:] = fftpack.diff(u2[i,:],period=py)
    d23[i,:] = fftpack.diff(u3[i,:],period=py)
    w23[i,:] = d23[i,:]
  d33[:,:] = -(d11[:,:]+d22[:,:])
  return

@jit
def part1(d11,d12,d13,d22,d23,d33,u1,u2,u3,px,py,w12,w13,w23):
  for k in range(u1.shape[2]):
    for j in range(u1.shape[1]):
      d11[:,j,k] = fftpack.diff(u1[:,j,k],period=px)
      d12[:,j,k] = fftpack.diff(u2[:,j,k],period=px)
      w12[:,j,k] = d12[:,j,k]
      d13[:,j,k] = fftpack.diff(u3[:,j,k],period=px)
      w13[:,j,k] = d13[:,j,k]
    for i in range(u1.shape[0]):
      aux = fftpack.diff(u1[i,:,k],period=py)
      d12[i,:,k] = (d12[i,:,k]+aux)/2
      w12[i,:,k] = (w12[i,:,k]-aux)/2
      d22[i,:,k] = fftpack.diff(u2[i,:,k],period=py)
      d23[i,:,k] = fftpack.diff(u3[i,:,k],period=py)
      w23[i,:,k] = d23[i,:,k]
  return

#@jit
def part2(d13,d23,d33,u1,u2,u3,pz,w13,w23):
  for j in range(u1.shape[1]):
    for i in range(u1.shape[0]):
      aux=fftpack.diff(u1[i,j,:],period=pz)
      d13[i,j,:] = (d13[i,j,:]+aux)/2
      w13[i,j,:] = (w13[i,j,:]-aux)/2
      aux=fftpack.diff(u2[i,j,:],period=pz)
      d23[i,j,:] = (d23[i,j,:]+aux)/2
      w23[i,j,:] = (w23[i,j,:]-aux)/2
      d33[i,j,:] = fftpack.diff(u3[i,j,:],period=pz)

  return

def part2p(d13,d23,d33,u1,u2,u3,pz,w13,w23):
  for i in range(u1.shape[0]):
    aux=fftpack.diff(u1[i,:],period=pz)
    d13[i,:] = (d13[i,:]+aux)/2
    w13[i,:] = (w13[i,:]-aux)/2
    aux=fftpack.diff(u2[i,:],period=pz)
    d23[i,:] = (d23[i,:]+aux)/2
    w23[i,:] = (w23[i,:]-aux)/2
    d33[i,:] = fftpack.diff(u3[i,:],period=pz)

  return

if len(sys.argv) == 3:
  Fil = sys.argv[1]
  N = sys.argv[2]

  sizefile = os.path.getsize(Fil+'u1_N_'+N)

  nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="../../../")
  px = 2*np.pi*xDomainSize
  py = 2*np.pi*yDomainSize
  pz = 2*np.pi*zDomainSize

  if sizefile == nx*ny*nz*8 :
    dtype='float64'
  elif sizefile == nx*ny*nz*4 :
    dtype='float32'

  a = datetime.datetime.now().replace(microsecond=0)

  d11 = np.memmap(Fil+'D11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d12 = np.memmap(Fil+'D12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d13 = np.memmap(Fil+'D13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d22 = np.memmap(Fil+'D22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d23 = np.memmap(Fil+'D23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d33 = np.memmap(Fil+'D33_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')

  w12 = np.memmap(Fil+'W12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  w13 = np.memmap(Fil+'W13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  w23 = np.memmap(Fil+'W23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')

  u1 = np.memmap(Fil+'u1_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  u2 = np.memmap(Fil+'u2_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
  u3 = np.memmap(Fil+'u3_N_'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

elif len(sys.argv) == 4:
  nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="./")
  px = 2*np.pi*xDomainSize
  py = 2*np.pi*yDomainSize
  pz = 2*np.pi*zDomainSize

  sizefile = os.path.getsize(sys.argv[1])

  if sizefile == (nx+2)*ny*nz*8 :
    dtype='float64'
  elif sizefile == (nx+2)*ny*nz*4 :
    dtype='float32'



  d11 = np.memmap('D11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d12 = np.memmap('D12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d13 = np.memmap('D13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d22 = np.memmap('D22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d23 = np.memmap('D23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  d33 = np.memmap('D33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')

  w12 = np.memmap('W12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  w13 = np.memmap('W13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')
  w23 = np.memmap('W23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='w+')

  u1 = np.memmap(sys.argv[1],shape=(nx+2,ny,nz),order='F',dtype=dtype,mode='r')[:-2,:,:]
  u2 = np.memmap(sys.argv[2],shape=(nx+2,ny,nz),order='F',dtype=dtype,mode='r')[:-2,:,:]
  u3 = np.memmap(sys.argv[3],shape=(nx+2,ny,nz),order='F',dtype=dtype,mode='r')[:-2,:,:]


a = datetime.datetime.now().replace(microsecond=0)  
num_cores = cpu_count()
print('Performing x and y derivatives\n')
Parallel(n_jobs=num_cores)(delayed(part1p)(d11[:,:,k],d12[:,:,k],d13[:,:,k],d22[:,:,k],d23[:,:,k],d33[:,:,k]
        ,u1[:,:,k],u2[:,:,k],u3[:,:,k],px,py,w12[:,:,k],w13[:,:,k],w23[:,:,k]) for k in range(d11.shape[2]))

#part1(d11,d12,d13,d22,d23,d33,u1,u2,u3,px,py,w12,w13,w23)
print('Performing z derivatives\n')
#part2(d13,d23,d33,u1,u2,u3,pz,w13,w23)
Parallel(n_jobs=num_cores)(delayed(part2p)(d13[:,j,:],d23[:,j,:],d33[:,j,:]
        ,u1[:,j,:],u2[:,j,:],u3[:,j,:],pz,w13[:,j,:],w23[:,j,:]) for j in range(d11.shape[1]))


print('Done\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))
