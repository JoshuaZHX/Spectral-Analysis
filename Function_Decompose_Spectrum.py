""" 
Program : Decompose_Spectrum 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 
from Function_Intensity_Image import Intensity_Image 

########## ----- Function : Decompose_Spectrum ----- ########## 
# Input  : l = The wavelength vector in $m$. 
#          A = The amplitude vector around area of interest. 
#          list_average_l = The list of wavelength that we are interested. 
# Output : list_average_A = The list of N*N intensity data for each wavelength. 

def Decompose_Spectrum(l, A, list_average_l): 
    list_average_A = [] 
    for i in range(len(list_average_l)): 
        Intensity = Intensity_Image(l, A, list_average_l[i]) 
        list_average_A.append(Intensity) 
    return list_average_A 