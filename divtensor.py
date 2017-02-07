#!/usr/bin/env python
import numpy as np
import sys
import os
from numba import jit
from globalfile import readglobal
import datetime

@jit(cache=True)
def quickdiff(u1,k,axis):
  return np.fft.irfft(np.fft.rfft(u1,axis=axis)*1j*k,axis=axis)

def diff(u1,length,axis):
  #length is the actual length divided by 2pi, which is what we get from the global file
  k1 = np.fft.rfftfreq(u1.shape[axis],d=length/u1.shape[axis])
  if (axis==0):
    aux1 = np.zeros(u1.shape[1])
    aux2 = np.zeros(u1.shape[2])
    k, aux1, aux1 = np.meshgrid(k1,aux1,aux2,indexing='ij')
  elif (axis==1):
    aux1 = np.zeros(u1.shape[0])
    aux2 = np.zeros(u1.shape[2])
    aux1, k, aux1 = np.meshgrid(aux1,k1,aux2,indexing='ij')
  elif (axis==2):
    aux1 = np.zeros(u1.shape[0])
    aux2 = np.zeros(u1.shape[1])
    aux1, aux1, k = np.meshgrid(aux1,aux2,k1,indexing='ij')
  del k1
  del aux1
  del aux2
  k = np.asfortranarray(k)
  return quickdiff(u1,k,axis)

a = datetime.datetime.now().replace(microsecond=0)

if os.path.exists('./global'): path='./'
elif os.path.exists('../global'): path='../'
elif os.path.exists('../../global'): path='../../'
elif os.path.exists('../../../global'): path='../../../'
elif os.path.exists('../../../../global'): path='../../../../'
elif os.path.exists('../../../../../global'): path='../../../../../'

nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path=path)
output = 'divtensor'

dtype='float64'
t11 = np.reshape(np.fromfile("t11",dtype=dtype),(nx,ny,nz),order='F')
t13 = np.reshape(np.fromfile("t13",dtype=dtype),(nx,ny,nz),order='F')

print("Calculating f1")
t11[:,:,:] = diff(t11,xDomainSize,0)
t11[:,:,:] = t11 + diff(t13,zDomainSize,2)

del t13

t12 = np.reshape(np.fromfile("t12",dtype=dtype),(nx,ny,nz),order='F')

t11[:,:,:] = t11 + diff(t12,yDomainSize,1)
t11.T.tofile(output+"1")
del t11
print("Calculating f2")

t22 = np.reshape(np.fromfile("t22",dtype=dtype),(nx,ny,nz),order='F')

t22[:,:,:] = diff(t22,yDomainSize,1)
t22[:,:,:] = t22 + diff(t12,xDomainSize,0)
del t12

t23 = np.reshape(np.fromfile("t23",dtype=dtype),(nx,ny,nz),order='F')

t22[:,:,:] = t22 + diff(t23,zDomainSize,2)
t22.T.tofile(output+"2")
del t22
print("Calculating f3")

t33 = np.reshape(np.fromfile("t33",dtype=dtype),(nx,ny,nz),order='F')

t33[:,:,:] = diff(t33,zDomainSize,2)
t33[:,:,:] = t33 + diff(t23,yDomainSize,1)
del t23

t13 = np.reshape(np.fromfile("t13",dtype=dtype),(nx,ny,nz),order='F')

t33[:,:,:] = t33 + diff(t13,xDomainSize,0)
t33.T.tofile(output+"3")

print('Done\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))

