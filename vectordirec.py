#!/usr/bin/env python
import numpy as np
import sys
import os
from globalfile import readglobal
import subprocess

if os.path.exists('./global'): path='./'
elif os.path.exists('../global'): path='../'
elif os.path.exists('../../global'): path='../../'
elif os.path.exists('../../../global'): path='../../../'
elif os.path.exists('../../../../global'): path='../../../../'
elif os.path.exists('../../../../../global'): path='../../../../../'

nx,ny,nz,xDomainSize,yDomainSize,zDomainSize = readglobal(path=path)

files = sys.argv[1:4]
output = sys.argv[4:]

subprocess.call(["truncate","-s",str(nx*ny*nz)]+output)
subprocess.call(["vectordir",str(nx*ny*nz)]+files+output)
