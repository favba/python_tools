#!/home/felipe_a/miniconda3#!/bin/python
import numpy as np
import sys
import os
from numba import jit
from globalfile import readglobal
import datetime

@jit(cache=True)
def filterbig(u1,x,y,z,boxdim,outputname,N,pathoutput,fil='G',dtype='float64'):
  if fil=='G':

    kx2 = (np.fft.fftfreq(u1.shape[0],d=2*np.pi*x/u1.shape[0]))**2
    ky2 = (np.fft.fftfreq(u1.shape[1],d=2*np.pi*y/u1.shape[1]))**2
    kz2 = (np.fft.rfftfreq(u1.shape[2],d=2*np.pi*z/u1.shape[2]))**2
    kx2, ky2, kz2 = np.meshgrid(kx2,ky2,kz2,indexing='ij')
    kx2+=ky2
    kx2+=kz2
    kx2 = np.asfortranarray(kx2)
    u1 = np.fft.irfftn(np.fft.rfftn(u1)*np.exp(-(kx2*(np.pi*boxdim)**2)/6))
    u1.T.tofile(pathoutput+"/"+fil+outputname+'_N'+N)

  return

a = datetime.datetime.now().replace(microsecond=0)

inputfile = sys.argv[1]
ngridpnts = sys.argv[2]
fil = sys.argv[3]

nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal("../../../")

sizefile = os.path.getsize(inputfile)

if sizefile == (nx+2)*ny*nz*8 :
  dtype = 'float64'
  padded = True
elif sizefile == nx*ny*nz*8:
  dtype = 'float64'
  padded = False
elif sizefile == (nx+2)*ny*nz*4 : 
  dtype = 'float32'
  padded = True
elif sizefile == nx*ny*nz*4:
  dtype = 'float32'
  padded = False
else : 
  print('Input file "'+inputfile+'" not matching number of gridpoints on global file')
  sys.exit(1)

if padded: u1=np.reshape(np.fromfile(inputfile,dtype=dtype),(nx+2,ny,nz),order='F')[:-2,:,:]
elif not padded : u1=np.reshape(np.fromfile(inputfile,dtype=dtype),(nx,ny,nz),order='F')

if fil == 'G':
  pathoutput = './Filtered_Fields/Gaussian/N'+ngridpnts
elif fil =='C':
  pathoutput = './Filtered_Fields/CutOff/N'+ngridpnts

if not os.path.exists(pathoutput): os.makedirs(pathoutput)

boxdim = np.float(ngridpnts)*2*zDomainSize*np.pi/nz
try: 
  outputname = inputfile[:inputfile.index(".")]
except: 
  outputname = inputfile

emantuptuo = outputname[::-1]

try:
  outputname = emantuptuo[:emantuptuo.index("/")][::-1]
except:
  outputname = outputname

u1 = u1.astype('float64')
print('Filtering File '+inputfile+'\n')
filterbig(u1, xDomainSize, yDomainSize, zDomainSize, boxdim, outputname, ngridpnts, pathoutput,fil=fil,dtype='float64')
print('Done\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))
