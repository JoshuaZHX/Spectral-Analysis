""" 
Program : Convert_Spectrum 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Convert_Spectrum ----- ########## 
# Input  : f     = The frequency vector in $s^{-1}$. 
#          A     = The 3D intensity vector wrt to frequency. 
#          sum_A = The 1D intensity vector wrt to frequency. 
#          left  = The wavelength at left extremity in $m$. 
#          right = The wavelength at right extremity in $m$. 
# Output : l     = The wavelength vector in $m$. 
#          A     = The 3D intensity vector. 
#          sum_A = The 1D intensity vector wrt to wavelength, normalized. 

def Convert_Spectrum(f, A, sum_A, left, right): 
    # Keep the positive-frequency section. Avoid dividing by zero. 
    index = np.where(f > 0)[0][0] 
    f     = f[index:] 
    A     = A[:, :, index:] 
    sum_A = sum_A[index:] 
    
    # keep the region of interest. 
    l           = (3e8 / f) 
    index_left  = np.where(l > right)[0][-1] 
    index_right = np.where(l < left)[0][0] 
    f           = f[index_left : index_right] 
    l           = l[index_left : index_right] 
    A           = A[:, :, index_left : index_right] 
    sum_A       = sum_A[index_left : index_right] 
    
    # Remove the background of the 1D intensity vector. 
    sum_A = sum_A - np.min(sum_A) 
    
    # Normalize the 1D intensity vector. 
    sum_A = sum_A / np.max(sum_A) 
    
    # Transform to the 1D intensity vector wrt to wavelength. 
    sum_A_l = (2*np.pi*3e8 / l**2) * sum_A 
    sum_A_l = sum_A_l / np.max(sum_A_l) 
    
    return f, l, A, sum_A, sum_A_l 