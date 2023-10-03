""" 
Program : Phase_Decomposition 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Phase_Decomposition ----- ########## 
# Input  : phase  = The unwrapped phase needed to be decomposed.   (N*N Grid) 
#          x      = The xy-axis.                                   (An Array) 
#          x0, y0 = The center of the phase in $m$.                (A Number) 
#          sigma  = The size of the phase in $m$.                  (A Number)         
# Output : coefficient = The coefficients of aberrations.          (A list) 

def Phase_Decomposition(phase, x, x0, y0, sigma): 
    X, Y   = np.meshgrid(x, x) 
    radius = np.sqrt((X - x0)**2 + (Y - y0)**2) / (2*np.sqrt(2)*sigma) 
    theta  = np.arctan2((Y - y0), (X - x0)) 
    
    Z1 = 2 * radius * np.sin(theta)                                # Tilt Vertical. {-1, 1} 
    Z2 = 2 * radius * np.cos(theta)                                # Tilt Horizontal. {+1, 1} 
    Z3 = np.sqrt(6) * (radius**2) * np.sin(2*theta)                # Astigmatism Oblique. {-2, 2} 
    Z4 = np.sqrt(6) * (radius**2) * np.cos(2*theta)                # Astigmatism Vertical. {+2, 2} 
    Z5 = np.sqrt(3) * (2*(radius**2) - 1)                          # Defocus. {0, 2} 
    Z6 = np.sqrt(8) * (3*(radius**3) - 2*radius) * np.sin(theta)   # Coma Vertical. {-1, 3} 
    Z7 = np.sqrt(8) * (3*(radius**3) - 2*radius) * np.cos(theta)   # Coma Horizontal. {+1, 3} 
    Z8 = np.sqrt(5) * (6*(radius**4) - 6*(radius**2) + 1)          # Spherical. {0, 4} 
    
    aberration  = [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8] 
    coefficient = [] 
    for Z in aberration: 
        Z[radius > 1] = 0 
        C = np.sum(phase * Z) / np.sum(Z**2) 
        coefficient.append(C) 
    
    return coefficient 