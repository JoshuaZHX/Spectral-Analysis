""" 
Program : Get_Data 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Get_Data ----- ########## 
# Input  : data  = The 'data' from h5 file. 
#          index = The index n,0,p.                 (A String) 
# Output : total = The sum intensity at each tau.   (An Array) 

def Get_Data(data, index): 
    plane = np.asarray(data[f'Sxytau_{index}'])[:, :, :] 
    Nt    = np.size(plane, axis=2) 
    total = np.zeros(Nt) 
    for i in range(Nt): 
        total[i] = np.sum(plane[:, :, i]) 
    return total 