"""
Functions for filtering data
"""

from numba import jit
import datetime
import numpy as np
import sys
from tempfile import TemporaryFile
from joblib import Parallel, delayed  
import multiprocessing

@jit
def dftxzC(u1,fouttemp):
  for j in range(u1.shape[1]) :
    fouttemp[:,j,:]=np.fft.fftn(u1[:,j,:],axes=(-1,-2)) 
  return  

@jit
def dftzG(u1,foutput,gfunc):
  for j in range(u1.shape[1]) :
    foutput[:,j,:] = np.fft.irfft(np.fft.rfft(u1[:,j,:],axis=-1)*gfunc,axis=-1)
  return  

def pdftzG(u1,foutput,gfunc):
  foutput[:,:] = np.fft.irfft(np.fft.rfft(u1[:,:],axis=-1)*gfunc,axis=-1)
  return  

@jit
def backC(fouttemp,foutput):
  for j in range(fouttemp.shape[1]) :
    foutput[:,j,:]=np.real(np.fft.ifftn(fouttemp[:,j,:],axes=(-1,-2)))
  return


def pdftxyG(foutput,gfunc):
  foutput[:,:] = np.fft.irfft2(np.fft.rfft2(foutput[:,:],axes=(-1,-2))*gfunc,axes=(-1,-2))
  return


def dftxyC(fouttemp,foutput,kx2,kz2,boxdim):

  print('Performing DFTs on xy slices.')
  for k in range(fouttemp.shape[2]) :
    kx2 += kz2[k]
    func= np.absolute(kx2) < (np.pi/boxdim)**2
    fouttemp[:,:,k] =np.fft.ifft(np.fft.fft(fouttemp[:,:,k],axis=1)*func,axis=1)

  print('\n')    
  print('Transforming back to real space...')
  back(fouttemp,foutput)

  return  


def filterbig(u1,x,y,z,boxdim,outputname,N,fil='G',dtype='float64'):

 
  num_cores = multiprocessing.cpu_count()

  if fil=='G':

    foutput = np.memmap('./Filtered_Fields/Gaussian/N'+N+'/'+fil+outputname+'_N_'+N,shape=(u1.shape[0],u1.shape[1],u1.shape[2]),dtype=dtype,order='F',mode='w+')


    print('Calculating Filtered Field...\n')
  
    a = datetime.datetime.now().replace(microsecond=0)

    kx2 = (np.fft.fftfreq(u1.shape[0],d=2*np.pi*x/u1.shape[0]))**2
    ky2 = (np.fft.fftfreq(u1.shape[1],d=2*np.pi*y/u1.shape[1]))**2
    kz2 = (np.fft.rfftfreq(u1.shape[2],d=2*np.pi*z/u1.shape[2]))**2
    kx2, kz2 = np.meshgrid(kx2,kz2,indexing='ij')
    del kx2
    kx2 = (np.fft.rfftfreq(u1.shape[0],d=2*np.pi*x/u1.shape[0]))**2
    kx2, ky2 = np.meshgrid(kx2,ky2,indexing='ij')

    kx2+=ky2
    del ky2
    kx2 = np.asfortranarray(kx2)
    kz2 = np.asfortranarray(kz2)
    print('Performing DFTs on yz slices.')
    
    gfunc = np.exp(-(kz2*(np.pi*boxdim)**2)/6)

#    dftzG(u1,foutput,gfunc)

    Parallel(n_jobs=num_cores)(delayed(pdftzG)(u1[:,j,:],foutput[:,j,:],gfunc) for j in range(u1.shape[1])) 


    del gfunc

    print('\n')
    

    gfunc = np.exp(-(kx2*(np.pi*boxdim)**2)/6)

    print('Performing DFTs on xy slices.')
 
    Parallel(n_jobs=num_cores)(delayed(pdftxyG)(foutput[:,:,k],gfunc) for k in range(u1.shape[2])) #, backend="threading"
    
    print('\nDone\n')
    b = datetime.datetime.now().replace(microsecond=0)
    print('Total time: {}'.format(b-a))


  elif fil=='C':
    foutput = np.memmap('./Filtered_Fields/CutOff/N'+N+'/'+fil+outputname+'_N_'+N,shape=(u1.shape[0],u1.shape[1],u1.shape[2]),dtype=dtype,order='F',mode='w+')


    print('Calculating Filtered Field...\n')
  
    a = datetime.datetime.now().replace(microsecond=0)

    kx2 = (np.fft.fftfreq(u1.shape[0],d=2*np.pi*x/u1.shape[0]))**2
    ky2 = (np.fft.fftfreq(u1.shape[1],d=2*np.pi*y/u1.shape[1]))**2
    kz2 = (np.fft.rfftfreq(u1.shape[2],d=2*np.pi*z/u1.shape[2]))**2
    kx2, kz2 = np.meshgrid(kx2,kz2,indexing='ij')
    del kx2
    kx2 = (np.fft.rfftfreq(u1.shape[0],d=2*np.pi*x/u1.shape[0]))**2
    kx2, ky2 = np.meshgrid(kx2,ky2,indexing='ij')

    kx2+=ky2
    del ky2
    kx2 = np.asfortranarray(kx2)
    kz2 = np.asfortranarray(kz2)
    print('Performing DFTs on yz slices.')
    
#    gfunc = np.exp(-(kz2*(np.pi*boxdim)**2)/6)
    gfunc = (np.absolute(kz2) < (np.pi/boxdim)**2)*1
#    dftzG(u1,foutput,gfunc)

    Parallel(n_jobs=num_cores)(delayed(pdftzG)(u1[:,j,:],foutput[:,j,:],gfunc) for j in range(u1.shape[1])) 


    del gfunc

    print('\n')
    

#    gfunc = np.exp(-(kx2*(np.pi*boxdim)**2)/6)
    gfunc = (np.absolute(kx2) < (np.pi/boxdim)**2)*1

    print('Performing DFTs on xy slices.')
 
    Parallel(n_jobs=num_cores)(delayed(pdftxyG)(foutput[:,:,k],gfunc) for k in range(u1.shape[2])) #, backend="threading"
    
    print('\nDone\n')
    b = datetime.datetime.now().replace(microsecond=0)
    print('Total time: {}'.format(b-a))

  return
