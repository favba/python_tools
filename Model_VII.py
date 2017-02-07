#!/usr/bin/env python
import numpy as np
import sys
from globalfile import readglobal
import datetime
from joblib import Parallel, delayed  
import multiprocessing
import numba as nb
import os
from decomp import inphdecomp

@nb.jit(cache=True,nopython=True)
def pf(t11,t22,t33,t12,t13,t23,
       II11,II22,II33,II12,II13,II23,
       d12,d13,d23,d11,d22,d33,
       n12,n13,n23,n11,n22,n33,
       index2,tm11,tm22,tm33,tm12,tm13,tm23):

  Nx, Ny = d11.shape
  hpi=2/np.pi
  
  for j in range(Ny):
    for i in range(Nx):
      
      b12=t12[i,j]
      b13=t13[i,j]
      b23=t23[i,j]
      b11=(2/3)*t11[i,j] - (1/3)*(t22[i,j]+t33[i,j])
      b22=(2/3)*t22[i,j] - (1/3)*(t11[i,j]+t33[i,j])
      b33=-b11-b22
    
      e12=d12[i,j]
      e13=d13[i,j]
      e23=d23[i,j]
      e11=d11[i,j]
      e22=d22[i,j]
      e33=d33[i,j]

      p12=n12[i,j]
      p13=n13[i,j]
      p23=n23[i,j]
      p11=n11[i,j]
      p22=n22[i,j]
      p33=n33[i,j]

      tp11 = b11 - II11[i,j]
      tp12 = b12 - II12[i,j]
      tp13 = b13 - II13[i,j]
      tp22 = b22 - II22[i,j]
      tp23 = b23 - II23[i,j]
      tp33 = b33 - II33[i,j]
    
      tm11p,tm22p,tm33p,tm12p,tm13p,tm23p,aux,aux,aux = inphdecomp(tp11,tp22,tp33,tp12,tp13,tp23,p11,p22,p33,p12,p13,p23)
 
      tm11[i,j],tm22[i,j],tm33[i,j],tm12[i,j],tm13[i,j],tm23[i,j],aux,aux,aux = inphdecomp(tm11p,tm22p,tm33p,tm12p,tm13p,tm23p,e11,e22,e33,e12,e13,e23)
 

      tm12[i,j]= tm12p - tm12[i,j] + II12[i,j]
      tm13[i,j]= tm13p - tm13[i,j] + II13[i,j]
      tm23[i,j]= tm23p - tm23[i,j] + II23[i,j]
      tm11[i,j]= tm11p - tm11[i,j] + II11[i,j]
      tm22[i,j]= tm22p - tm22[i,j] + II22[i,j]
      tm33[i,j]= tm33p - tm33[i,j] + II33[i,j]

      modm=np.sqrt((tm11[i,j]**2 + 2*tm12[i,j]**2 + 2*tm13[i,j]**2 + tm22[i,j]**2 + 2*tm23[i,j]**2 + tm33[i,j]**2))
      modb=np.sqrt((b11**2 + 2*b12**2 + 2*b13**2 + b22**2 + 2*b23**2 + b33**2))

      index2[i,j] = 1 - hpi*np.arccos(modm/modb)
      # index2[i,j] = modm/modb
 
  return


Fil = sys.argv[1]
N = sys.argv[2]
nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path="../../../")

dtype = 'Float64'
if os.path.getsize(Fil+'P11_N'+N) == nx*ny*nz*4 : dtype = 'Float32'


d11 = np.memmap(Fil+'D11_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d12 = np.memmap(Fil+'D12_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d13 = np.memmap(Fil+'D13_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d22 = np.memmap(Fil+'D22_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d23 = np.memmap(Fil+'D23_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
d33 = np.memmap(Fil+'D33_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')

p11 = np.memmap(Fil+'P11_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
p12 = np.memmap(Fil+'P12_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
p13 = np.memmap(Fil+'P13_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
p22 = np.memmap(Fil+'P22_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
p23 = np.memmap(Fil+'P23_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')
p33 = np.memmap(Fil+'P33_N'+N,shape=(nx,ny,nz),order='F',dtype = dtype,mode='r')

t11 = np.memmap(Fil+'T11_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t12 = np.memmap(Fil+'T12_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t13 = np.memmap(Fil+'T13_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t22 = np.memmap(Fil+'T22_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t23 = np.memmap(Fil+'T23_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
t33 = np.memmap(Fil+'T33_N'+N,shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

II11 = np.memmap('./Model_II/Tm11',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
II12 = np.memmap('./Model_II/Tm12',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
II13 = np.memmap('./Model_II/Tm13',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
II22 = np.memmap('./Model_II/Tm22',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
II23 = np.memmap('./Model_II/Tm23',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')
II33 = np.memmap('./Model_II/Tm33',shape=(nx,ny,nz),order='F',dtype=dtype,mode='r')

pathoutput = './Model_VII'

if not os.path.exists(pathoutput): os.makedirs(pathoutput)

index2 = np.memmap('./Model_VII/index7',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm11 = np.memmap('./Model_VII/Tm11',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm12 = np.memmap('./Model_VII/Tm12',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm13 = np.memmap('./Model_VII/Tm13',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm22 = np.memmap('./Model_VII/Tm22',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm23 = np.memmap('./Model_VII/Tm23',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')
tm33 = np.memmap('./Model_VII/Tm33',shape=(nx,ny,nz),order='F',dtype = dtype,mode='w+')

a = datetime.datetime.now().replace(microsecond=0)
num_cores = multiprocessing.cpu_count()

print('Performing Calculations in {} cores'.format(num_cores))

Parallel(n_jobs=num_cores)(delayed(pf)(
       t11[:,:,k],t22[:,:,k],t33[:,:,k],t12[:,:,k],t13[:,:,k],t23[:,:,k],
       II11[:,:,k],II22[:,:,k],II33[:,:,k],II12[:,:,k],II13[:,:,k],II23[:,:,k],
       d12[:,:,k],d13[:,:,k],d23[:,:,k],d11[:,:,k],d22[:,:,k],d33[:,:,k],
       p12[:,:,k],p13[:,:,k],p23[:,:,k],p11[:,:,k],p22[:,:,k],p33[:,:,k],
       index2[:,:,k],tm11[:,:,k],tm22[:,:,k],tm33[:,:,k],tm12[:,:,k],tm13[:,:,k],tm23[:,:,k]) for k in range(nz))  

print('Done\n')
b = datetime.datetime.now().replace(microsecond=0)
print('Total time: {}\n'.format(b-a))
