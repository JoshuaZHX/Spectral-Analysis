""" 
Program : Find_Centroid 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Find_Centroid ----- ########## 
# Input  : x         = The xy-axis of intensity data. 
#          Intensity = The intensity data. 
# Output : x0        = The x-coordinate of the center. 
#          y0        = The y-coordinate of the center. 

def Find_Centroid(x, Intensity): 
    x_sum = np.sum(Intensity, axis=0) 
    y_sum = np.sum(Intensity, axis=1) 
    x0    = np.sum(x * x_sum) / np.sum(x_sum) 
    y0    = np.sum(x * y_sum) / np.sum(y_sum) 
    return x0, y0 