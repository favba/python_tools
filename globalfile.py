"""
Get information from Global file
"""
def readglobal(path="./",tell=True):
  
  if tell : print('\nReading global file...\n')
  with open(path+'global','r') as f:
    globalfile=f.read()
    
  index = globalfile.find('nx')
  indexend=globalfile[index:].find('\n')
  nx = int(globalfile[index+len('nx')+1:index+indexend])
    
  index = globalfile.find('ny')
  indexend=globalfile[index:].find('\n')
  ny = int(globalfile[index+len('ny')+1:index+indexend])
    
  index = globalfile.find('nz')
  indexend=globalfile[index:].find('\n')
  nz = int(globalfile[index+len('nz')+1:index+indexend])
    
  index = globalfile.find('xDomainSize')
  indexend=globalfile[index:].find('\n')
  xDomainSize = float(globalfile[index+len('xDomainSize')+1:index+indexend])
    
  index = globalfile.find('yDomainSize')
  indexend=globalfile[index:].find('\n')
  yDomainSize = float(globalfile[index+len('yDomainSize')+1:index+indexend])
    
  index = globalfile.find('zDomainSize')
  indexend=globalfile[index:].find('\n')
  zDomainSize = float(globalfile[index+len('zDomainSize')+1:index+indexend])
  
  if tell : print('Nx: {}\nNy: {}\nNz: {}\nxDomainSize: {}\nyDomainSize: {}\nzDomainSize: {}\n'.format(
    nx,ny,nz,xDomainSize,yDomainSize,zDomainSize))
  return nx,ny,nz,xDomainSize,yDomainSize,zDomainSize
