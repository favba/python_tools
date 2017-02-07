"""
Compiled functions with parallel loops
"""
import numpy as np
import numba as nb
from joblib import Parallel, delayed 
import multiprocessing
from tempfile import TemporaryFile

@nb.jit(nogil=True)
def maxminmeanc(field,index=False):
  nx,ny = field.shape
  emin = field[0,0]
  emax = field[0,0]
  emean = 0
  if index:
    for j in range(ny):
      emeanx = 0.0
      for i in range(nx):
        if field[i,j] <= emin :
          emin = field[i,j]
          imin = i
          jmin = j
        elif field[i,j] >= emax : 
          emax = field[i,j]
          imax = i
          jmax = j
        emeanx += field[i,j]
      emean += emeanx/nx
    return emax, emin, emean/ny, imax, jmax, imin, jmin
  else:
    for j in range(ny):
      emeanx = 0.0
      for i in range(nx):
        if field[i,j] <= emin : emin = field[i,j]
        elif field[i,j] >= emax : emax = field[i,j]
        emeanx += field[i,j]
      emean += emeanx/nx
    return emax, emin, emean/ny

def maxminmean(field, num_cores=multiprocessing.cpu_count(),index=False):
  """
  Return the max, min and mean of array
  """
  nx,ny,nz = field.shape
  
  num_cores = multiprocessing.cpu_count()
  r = np.array(Parallel(n_jobs=num_cores,backend='threading')(delayed(maxminmeanc)(field[:,:,k],index) for k in range(nz)))

  maxv = np.max(r[:,0])
  minv = np.min(r[:,1])
  mean = np.mean(r[:,2])
  
  if index:
    kmax = np.where(r[:,0] == maxv)[0][0]
    imax = r.astype('int')[kmax,3]
    jmax = r.astype('int')[kmax,4]

    kmin = np.where(r[:,1] == minv)[0][0]
    imin = r.astype('int')[kmin,5]
    jmin = r.astype('int')[kmin,6]
    
    return maxv, minv, mean, imax, jmax, kmax, imin, jmin, kmin
  else: return maxv, minv, mean


@nb.jit(nogil=True)
def maxminc(field):
  nx = len(field)
  
  if np.remainder(nx,2) == 0:
    if field[0] < field[1]:
      emin = field[0]
      emax = field[1]
    else:
      emin = field[1]
      emax = field[0]
    init = 2
  else:
    emin = field[0]
    emax = field[0]
    init = 1

  for i in range(init,nx-1,2):
    if field[i] >= field[i+1]:
      if field[i] > emax : emax = field[i]
      if field[i+1] < emin: emin = field[i+1]
    else :
      if field[i+1] > emax : emax = field[i+1]
      if field[i] < emin: emin = field[i]
  
  return emax, emin

def maxmin(field, num_cores=multiprocessing.cpu_count()):
  """
  Return the max and min of array
  """
  if len(field.shape) == 1:
    maxv = np.max(field)
    minv = np.min(field)
  
  else:
    nx,ny,nz = field.shape

    r = np.array(Parallel(n_jobs=num_cores,backend='threading')(delayed(maxminc)(np.ravel(field[:,:,k],order='F')) for k in range(nz)))

    maxv = np.max(r[:,0])
    minv = np.min(r[:,1])
  
  return maxv, minv

@nb.jit(nogil=True)
def histc(field,binvals,result):
  binss  = len(binvals)
  j = 0

  for i in range(binss):
    result[i] = 0
    while j < len(field) and field[j] <= binvals[i]:
      result[i] += 1
      j += 1
#    print(result[i])
  return

def histc1(field,binvals,result):
  histc(field,binvals,result)
  return

def hist(field,bins = 200, num_cores=multiprocessing.cpu_count(),density=None):
  
  maxv, minv = maxmin(field,num_cores)
 
  dx = (maxv-minv)#!/bins

  binvals = np.linspace(minv+dx,maxv,bins)
  binEdges = np.linspace(minv,maxv,bins+1)

  nx,ny,nz = field.shape

  presults = np.memmap(TemporaryFile(),shape=(bins,nz), order = 'F', dtype = np.int,mode='w+')
  presults[:,:] = 0
  Parallel(n_jobs=num_cores,backend='threading')(delayed(histc)(np.sort(field[:,:,k],axis=None),binvals,presults[:,k]) for k in range(nz)) #

  a = np.sum(presults, axis = 1)
  del presults

  if density: a = a/np.trapz(a,dx = dx)
  
  return a, binEdges
