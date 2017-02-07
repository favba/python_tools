import numpy as np
import numba as nb

@nb.jit(nopython=True,cache=True)
def eigen(e11,e22,e33,e12,e13,e23):

  p1 = e12**2 + e13**2 + e23**2
  if (p1 == 0.0):
    # E is diagonal.
    eigvE1 = e11
    eigvE2 = e22
    eigvE3 = e33

    eigvecE11 = 1.0
    eigvecE12 = 0.0
    eigvecE13 = 0.0

    eigvecE21 = 0.0
    eigvecE22 = 1.0
    eigvecE23 = 0.0

    eigvecE31 = 0.0
    eigvecE32 = 0.0
    eigvecE33 = 1.0

    return eigvE1, eigvE2, eigvE3, eigvecE11, eigvecE12, eigvecE13, eigvecE21, eigvecE22, eigvecE23, eigvecE31, eigvecE32, eigvecE33

  else:
    q = (e11 + e22 + e33)/3
    p2 = (e11-q)**2 + (e22-q)**2 + (e33-q)**2 + 2*p1
    p = np.sqrt(p2/6)
    r = ((e11-q)*(e22-q)*(e33-q) - (e11-q)*(e23**2) - (e12**2)*(e33-q) + 2*(e12*e13*e23) - (e13**2)*(e22-q))/(2*p*p*p)

    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.
    if (r <= -1): 
      phi = np.pi/3
    elif (r >= 1):
      phi = 0
    else:
      phi = np.arccos(r)/3

    # the eigenvalues satisfy eigvE3 <= eigvE2 <= eigvE1
    eigvE1 = q + 2*p*np.cos(phi)
    eigvE3 = q + 2*p*np.cos(phi+(2*np.pi/3))
    eigvE2 = 3*q - eigvE1 - eigvE3     # since trace(E) = eigvE1 + eigvE2 + eigvE3 = 3q


    bla = ((e22 - eigvE1)*(e33 - eigvE1) - e23*e23)
    if bla != 0.0: 
      eigvecE11 = 1.0
      eigvecE12 = (e23*e13 - (e33-eigvE1)*e12)/bla
      eigvecE13 = (-e13 -e23*eigvecE12)/(e33-eigvE1)
      aux = np.sqrt(1.0 + eigvecE12**2 + eigvecE13**2)
      eigvecE11 = 1/aux
      eigvecE12 = eigvecE12/aux
      eigvecE13 = eigvecE13/aux
    else:
      bla = ((e11 - eigvE1)*(e22 - eigvE1) - e12*e12)
      eigvecE13 = 1.0
      eigvecE11 = (e23*e12 - (e22-eigvE1)*e13)/bla
      eigvecE12 = (-e23 -e12*eigvecE11)/(e22-eigvE1)
      aux = np.sqrt(1.0 + eigvecE12**2 + eigvecE11**2)
      eigvecE11 = eigvecE11/aux
      eigvecE12 = eigvecE12/aux
      eigvecE13 = 1.0/aux

    bla = ((e22 - eigvE2)*(e33 - eigvE2) - e23*e23)
    if bla != 0.0 : 
      eigvecE21 = 1.0
      eigvecE22 = (e23*e13 - (e33-eigvE2)*e12)/bla
      eigvecE23 = (-e13 -e23*eigvecE22)/(e33-eigvE2)
      aux = np.sqrt(1.0 + eigvecE22**2 + eigvecE23**2)
      eigvecE21 = 1/aux
      eigvecE22 = eigvecE22/aux
      eigvecE23 = eigvecE23/aux
    else:
      bla = ((e11 - eigvE2)*(e22 - eigvE2) - e12*e12)
      eigvecE23 = 1.0
      eigvecE21 = (e23*e12 - (e22-eigvE2)*e13)/bla
      eigvecE22 = (-e23 -e12*eigvecE21)/(e22-eigvE2)
      aux = np.sqrt(1.0 + eigvecE22**2 + eigvecE21**2)
      eigvecE21 = eigvecE21/aux
      eigvecE22 = eigvecE22/aux
      eigvecE23 = 1.0/aux

    eigvecE31 = eigvecE12*eigvecE23 - (eigvecE13*eigvecE22)
    eigvecE32 = eigvecE13*eigvecE21 - (eigvecE11*eigvecE23)
    eigvecE33 = eigvecE11*eigvecE22 - (eigvecE12*eigvecE21)

    return eigvE1, eigvE2, eigvE3, eigvecE11, eigvecE12, eigvecE13, eigvecE21, eigvecE22, eigvecE23, eigvecE31, eigvecE32, eigvecE33



@nb.jit(nopython=True,cache=True)
def inphdecomp(t11,t22,t33,t12,t13,t23,
               e11,e22,e33,e12,e13,e23):

    (eigvE1, eigvE2, eigvE3, 
    eigvecE11, eigvecE12, eigvecE13, 
    eigvecE21, eigvecE22, eigvecE23, 
    eigvecE31, eigvecE32, eigvecE33) = eigen(e11,e22,e33,e12,e13,e23)

    m1 = t11*eigvecE11*eigvecE11 + t22*eigvecE12*eigvecE12 + t33*eigvecE13*eigvecE13 + 2*(
         t12*eigvecE11*eigvecE12 + t13*eigvecE11*eigvecE13 + t23*eigvecE12*eigvecE13)
    m2 = t11*eigvecE21*eigvecE21 + t22*eigvecE22*eigvecE22 + t33*eigvecE23*eigvecE23 + 2*(
         t12*eigvecE21*eigvecE22 + t13*eigvecE21*eigvecE23 + t23*eigvecE22*eigvecE23)
    m3 = -m1 -m2

    # modm=np.sqrt(m1**2 + m2**2 + m3**2)
    # modb=np.sqrt(t11*t11 + t22*t22 + t33*t33 + 2*(t12*t12 + t13*t13 + t23*t23))
    # index = 1 - hpi*np.arccos(modm/modb)

    lam2 = eigvE1**2 + eigvE2**2 + eigvE3**2

    a = 1 - (3*eigvE1**2/lam2)
    b = 1 - (3*eigvE2**2/lam2)
    c = b*eigvE1 - a*eigvE2

    alpha1 = (b*m1 - a*m2)/c
    alpha0 = (m1 - eigvE1*alpha1)/a
    alpha2 = -3*alpha0/lam2

    tm11 = m1*eigvecE11*eigvecE11 + m2*eigvecE21*eigvecE21 + m3*eigvecE31*eigvecE31
    tm22 = m1*eigvecE12*eigvecE12 + m2*eigvecE22*eigvecE22 + m3*eigvecE32*eigvecE32
    #      tm33 = m1*eigvecE13*eigvecE13 + m2*eigvecE23*eigvecE23 + m3*eigvecE33*eigvecE33
    tm33 = -tm11 -tm22  
    tm12 = m1*eigvecE11*eigvecE12 + m2*eigvecE21*eigvecE22 + m3*eigvecE31*eigvecE32
    tm13 = m1*eigvecE11*eigvecE13 + m2*eigvecE21*eigvecE23 + m3*eigvecE31*eigvecE33
    tm23 = m1*eigvecE12*eigvecE13 + m2*eigvecE22*eigvecE23 + m3*eigvecE32*eigvecE33

    return tm11,tm22,tm33,tm12,tm13,tm23,alpha0,alpha1,alpha2


@nb.jit(nopython=True,cache=True)
def propdecomp(t11,t22,t33,t12,t13,t23,
               e11,e22,e33,e12,e13,e23):

    mode2 = (e11**2 + 2*e12**2 + 2*e13**2 + e22**2 + 2*e23**2 + e33**2)
    alphat = (e11*t11 + 2*e12*t12 + 2*e13*t13 + e22*t22 + 2*e23*t23 + e33*t33)/mode2

    m12 = alphat*e12
    m13 = alphat*e13
    m23 = alphat*e23
    m11 = alphat*e11
    m22 = alphat*e22
    m33 = alphat*e33

    # modm=np.sqrt((m11**2 + 2*m12**2 + 2*m13**2 + m22**2 + 2*m23**2 + m33**2))
    # modb=np.sqrt((((2/3)*t11-(1/3)*(t22+t33))**2 + 2*t12**2 + 2*t13**2 + ((2/3)*t22-(1/3)*(t11+t33))**2 + 2*t23**2 + ((2/3)*t33-(1/3)*(t11+t22))**2))

    # index1 = 1 - hpi*np.arccos(modm/modb)

    return m11,m22,m33,m12,m13,m23,alphat
