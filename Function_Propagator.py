""" 
Program : Function_Propagator 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Propagator ----- ########## 
# Input  : E0 = The Gaussian beam at z=0.            (N*N Grid) 
#          l  = The wavelength in $m$.               (A Number) 
#          x  = The xy-axis.                         (An Array) 
#          D  = The propagation distance from z=0.   (A Number) 
# Output : E7 = The Gaussian beam at z=D.            (N*N Grid) 

def Propagator(E0, l, x, D): 
    # Compute kZ. 
    k      = (2*np.pi) / l 
    dx     = x[1] - x[0] 
    N      = np.size(x) 
    F      = 1/dx 
    df     = F/N 
    kx     = np.linspace(-(N/2)*df*2*np.pi, (N/2)*df*2*np.pi, N) 
    kX, kY = np.meshgrid(kx, kx) 
    kZ     = np.real(np.emath.sqrt(k**2 - kX**2 - kY**2)) 
    
    # Propagation. 
    E1 = np.fft.fftshift(E0) 
    E2 = np.fft.fft2(E1) 
    E3 = np.fft.ifftshift(E2) 
    
    E4 = E3 * np.exp(1j*kZ*D) 
    
    E5 = np.fft.fftshift(E4) 
    E6 = np.fft.ifft2(E5) 
    E7 = np.fft.ifftshift(E6) 
    return E7 