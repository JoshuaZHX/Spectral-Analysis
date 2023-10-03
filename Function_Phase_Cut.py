""" 
Program : Phase_Cut
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Phase_Cut ----- ########## 
# Input  : phase  = The wrapped phase.                         (N*N Grid) 
#          x      = The xy-axis.                               (An Array) 
#          x0, y0 = The center of the phase in $m$.            (A Number) 
#          sigma  = The size of the phase in $m$.              (A Number) 
# Output : phase  = The wrapped phase in region of interest.   (N*N Grid) 


def Phase_Cut(phase, x, x0, y0, sigma): 
    X, Y   = np.meshgrid(x, x) 
    radius = np.sqrt((X - x0)**2 + (Y - y0)**2) / (np.sqrt(2)*sigma) 
    phase[radius > 1] = 0 
    return phase 