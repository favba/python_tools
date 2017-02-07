#!/usr/bin/env python
""" 
Evaluate planewise expressions of the type "f1+f2-(np.sqrt(f3)*f1**2)" where f$ is the $ file inputed.
The last argument must be the output file name and before it a string with the expression to be evaluated
"""

import numpy as np
import sys
from globalfile import readglobal
import os

if os.path.exists('./global'): path='./'
elif os.path.exists('../global'): path='../'
elif os.path.exists('../../global'): path='../../'
elif os.path.exists('../../../global'): path='../../../'
elif os.path.exists('../../../../global'): path='../../../../'
elif os.path.exists('../../../../../global'): path='../../../../../'
else: path = input('Please specify global file directory path: ')

nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path=path)


files_names = sys.argv[1:-2]
sizefile = [os.path.getsize(f) for f in files_names]

files = [None]*len(files_names)

for i in range(len(files_names)):

  if sizefile[i] == nx*ny*nz*8 :
    files[i] = np.memmap(files_names[i],shape=(nx,ny,nz),order='F',dtype='float64',mode='r')
  elif sizefile[i] == (nx+2)*ny*nz*8: 
    files[i] = np.memmap(files_names[i],shape=(nx+2,ny,nz),order='F',dtype='float64',mode='r')[:-2,:,:]
  elif sizefile[i] == nx*ny*nz*4 :
    files[i] = np.memmap(files_names[i],shape=(nx,ny,nz),order='F',dtype='float32',mode='r')
  elif sizefile[i] == (nx+2)*ny*nz*4: 
    files[i] = np.memmap(files_names[i],shape=(nx+2,ny,nz),order='F',dtype='float32',mode='r')[:-2,:,:]

#files=[np.memmap(f,shape=(nx,ny,nz),order='F',dtype='float64',mode='r') for f in files_names]  

expression = sys.argv[-2]
tell = sys.argv[-2]
for i in range(len(files_names)):
  ii=i+1
  expression = expression.replace('f'+str(ii),'files['+str(i)+'][:,:,k]')
  tell = tell.replace('f'+str(ii),files_names[i])


print('Performing Calculation: '+sys.argv[-1]+" = "+tell)
result=np.memmap(sys.argv[-1],shape=files[0].shape,order='F',dtype='float64',mode='w+')

codetoexec="for k in range(files[0].shape[2]): result[:,:,k] = expression".replace('expression',expression)
exec(codetoexec)  
print('Done')
