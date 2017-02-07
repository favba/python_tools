#!/usr/bin/env python
import sys
import os
import numpy as np
from numba import jit
import argparse
from globalfile import readglobal

@jit(cache=True)
def filterbig(u1,x,y,z,boxdim,fil='G',dtype='float64'):
  if fil=='G':

    kx2 = (np.fft.fftfreq(u1.shape[0],d=2*np.pi*x/u1.shape[0]))**2
    ky2 = (np.fft.fftfreq(u1.shape[1],d=2*np.pi*y/u1.shape[1]))**2
    kz2 = (np.fft.rfftfreq(u1.shape[2],d=2*np.pi*z/u1.shape[2]))**2
    kx2, ky2, kz2 = np.meshgrid(kx2,ky2,kz2,indexing='ij')
    kx2+=ky2
    kx2+=kz2
    kx2 = np.asfortranarray(kx2)
    u1[:,:,:] = np.fft.irfftn(np.fft.rfftn(u1)*np.exp(-(kx2*(np.pi*boxdim)**2)/6))

  return

parser = argparse.ArgumentParser(description='Generate smooth random field')
parser.add_argument("file", type = str, nargs='+', help = "Output files name")
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('-n','--normal', action = 'store_true', help = "Use normal distribution")
group.add_argument('-u','--uniform', action = 'store_true', help = "Use uniform distribution")


if os.path.exists('./global'): path='./'
elif os.path.exists('../global'): path='../'
elif os.path.exists('../../global'): path='../../'
elif os.path.exists('../../../global'): path='../../../'
elif os.path.exists('../../../../global'): path='../../../../'
elif os.path.exists('../../../../../global'): path='../../../../../'
else: path = input('Please specify global file directory path: ')

nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path=path,tell=False)

args = parser.parse_args()

boxdim = 6*2*zDomainSize*np.pi/nz

for output in args.file :
  a = np.asfortranarray(np.random.normal(size=(nx,ny,nz)))
  filterbig(a,xDomainSize,yDomainSize,zDomainSize,boxdim,fil='G',dtype='float64')
  if args.uniform : 
    b = np.asfortranarray(np.random.normal(size=(nx,ny,nz)))
    filterbig(b,xDomainSize,yDomainSize,zDomainSize,boxdim,fil='G',dtype='float64')
    c = np.asfortranarray(np.random.normal(size=(nx,ny,nz)))
    filterbig(c,xDomainSize,yDomainSize,zDomainSize,boxdim,fil='G',dtype='float64')
    a = a/np.sqrt(a*a+b*b+c*c)
  a.T.tofile(output)
