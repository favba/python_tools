#!/home/felipe_a/miniconda3/bin/python3
import numpy as np
import sys
from globalfile import readglobal
from filters import filterbig
import os

inputfile = sys.argv[1]
ngridpnts = sys.argv[2]
fil = sys.argv[3]

nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal()

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

if padded: u1=np.memmap(inputfile,shape=(nx+2,ny,nz),order='F',dtype=dtype,mode='r')[:-2,:,:]
elif not padded : u1=np.memmap(inputfile,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

if fil == 'G':
  pathoutput = './Filtered_Fields/Gaussian/N'+ngridpnts
elif fil =='C':
  pathoutput = './Filtered_Fields/CutOff/N'+ngridpnts

if not os.path.exists(pathoutput): os.makedirs(pathoutput)

boxdim = np.float(ngridpnts)*2*zDomainSize*np.pi/nz
try: outputname = inputfile[:inputfile.index(".")]
except: outputname = inputfile
filterbig(u1,xDomainSize,yDomainSize,zDomainSize,boxdim,outputname,ngridpnts,fil=fil,dtype=dtype)
