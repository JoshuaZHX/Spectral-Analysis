""" 
Program : Gerchberg-Saxton Algorithm 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 
from Function_GSA_Cycle import GSA_Cycle 
from Function_Phase_Generator import Phase_Generator 

########## ----- Function : Gerchberg-Saxton Algorithm ----- ########## 
# Input  : I2 = The intensity at z=n.                (N*N Grid) 
#          I1 = The intensity at z=0.                (N*N Grid) 
#          I3 = The intensity at z=p.                (N*N Grid) 
#          l  = The wavelength in $m$.               (A Number) 
#          x  = The xy-axis.                         (An Array) 
#          zR = The Rayleigh range in $m$.           (A Number) 
#          C  = The cofficients for initial phase.   (5 Number) 
#          D  = The propagation distance.            (A Number) 
#          iteration = The number of iterations.     (A Number) 
# Output : ph         = The recovered phase.               
#          I1_updated = The updataed normalized intensity at z=0. 
#          error1     = The error history at z=0. 

def Gerchberg_Saxton_Algorithm(I2, I1, I3, l, x, zR, C, D, iteration): 
    # Compute Initial Phase. 
    w0     = np.sqrt(zR*l / np.pi) 
    X, Y   = np.meshgrid(x, x) 
    radius = np.sqrt(X**2 + Y**2) / (2*w0) 
    theta  = np.arctan2(Y, X) 
    ph     = Phase_Generator(radius, theta, C[0], C[1], C[2], C[3], C[4]) 
    
    # Prepare Gerchberg-Saxton Algorithm. 
    I1     = I1 / np.max(I1) 
    I2     = I2 / np.max(I2) 
    I3     = I3 / np.max(I3) 
    A1     = np.sqrt(I1) 
    A2     = np.sqrt(I2) 
    A3     = np.sqrt(I3) 
    I1_sum = np.sum(I1) 
    I2_sum = np.sum(I2) 
    I3_sum = np.sum(I3) 
    error1 = np.zeros(iteration) 
    error2 = np.zeros(iteration) 
    error3 = np.zeros(iteration) 
    
    # The Gerchberg-Saxton Algorithm. 
    for i in range(iteration): 
        ph, I1_updated, I2_updated, I3_updated = GSA_Cycle(A1, A2, A3, ph, l, x, D) 
        I1_updated = I1_updated / np.max(I1_updated) 
        I2_updated = I2_updated / np.max(I2_updated) 
        I3_updated = I3_updated / np.max(I3_updated) 
        error1[i]  = np.sum(np.abs(I1 - I1_updated)) / I1_sum 
        error2[i]  = np.sum(np.abs(I2 - I2_updated)) / I2_sum 
        error3[i]  = np.sum(np.abs(I3 - I3_updated)) / I3_sum 
    
    return ph, I1_updated, I2_updated, I3_updated, error1, error2, error3 