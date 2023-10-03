""" 
Program : Get_Spectrum 
Author : Haoxuan ZHANG & Joséphine MONZAC 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Get_Spectrum ----- ########## 
# Input  : data     = The 'data' from h5 file. 
#          x        = The xy-axis in $m$. 
#          t        = The t-axis in $s$. 
#          index    = The index n,0,p.             (A String) 
#          zero_pad = The zero_padding scale.      (A Number) 
# Output : frequency_bis = The frequency vector with unit $s^{-1}$. 
#          TF_S_bis      = The 3D intensity vector. 
#          int_TF_S_bis  = The 1D intensity vector. 

# Warning : The size of TF_S_bis is Ns*Ns*N_bis. 

def Get_Spectrum(data, x, t, index, zero_pad): 
    plane = np.asarray(data[f'Sxytau_{index}'])[:, :, :] 
    Nx    = np.size(x)            # The number of points in xy-axis. 
    Nt    = np.size(t)            # The number of points in t-axis. 
    dt    = (t[1] - t[0])         # The size of one t-step with unit $s$. 
    
    # To improve the accuracy of numerical computation. 
    N_bis = zero_pad * Nt 
    S_bis = np.zeros((Nx, Nx, N_bis)) 
    S_bis[:, :, int((N_bis - Nt)/2):int((N_bis + Nt)/2)] = plane 
    
    frequency_bis = np.arange(-N_bis/2, N_bis/2)*1.0 / (dt * N_bis) 
    TF_S_bis      = np.abs(np.fft.fftshift(np.fft.fft(S_bis), axes=2)) / Nt 
    int_TF_S_bis  = np.zeros(N_bis) 
    for i in range(N_bis): 
        int_TF_S_bis[i] = np.sum(TF_S_bis[:, :, i]) 
    
    return frequency_bis, TF_S_bis, int_TF_S_bis 