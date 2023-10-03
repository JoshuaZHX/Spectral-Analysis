""" 
Program : Intensity_Image 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Intensity_Image ----- ########## 
# Input  : l          = The wavelength vector in $m$. 
#          A          = The 3D amplitude vector around area of interest. 
#          wavelength = The wavelength that we are interested. 
# Output : average_A  = The N*N intensity data for this wavelength. 

def Intensity_Image(l, A, wavelength): 
    upper_l    = wavelength + (5*1e-9) 
    lower_l    = wavelength - (5*1e-9) 
    upperindex = np.where(l >= upper_l)[0][-1] 
    lowerindex = np.where(l <= lower_l)[0][0] 
    average_A  = np.average(A[:, :, upperindex : lowerindex], axis = 2) 
    return average_A 