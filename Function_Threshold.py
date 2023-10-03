""" 
Program : Threshold 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Threshold ----- ########## 
# Input  : Intensity = The intensity data before threshold. 
#          value     = The threshold filtering value. 
# Output : Intensity = The intensity data after threshold. 

# Warning !!! Assumption : value < 1 

def Threshold(Intensity, value): 
    if (value >= 1): 
        raise ValueError 
    threshold = value * np.max(Intensity) 
    Intensity[Intensity <= threshold] = 0 
    return Intensity 