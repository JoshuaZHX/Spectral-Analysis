""" 
Program : Spectral_Analysis 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import time 
import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle 

# Intensity Data 
from Function_Get_Data import Get_Data 
from Function_Get_Spectrum import Get_Spectrum 
from Function_Convert_Spectrum import Convert_Spectrum 

# Gerchberg-Saxton Algorithm Preparation 
from Function_Phase_Generator import Phase_Generator 

# Spectral Analysis 
from Function_Threshold import Threshold 
from Function_Propagator import Propagator 
from Function_Find_Sigma import Find_Sigma 
from Function_Find_Centroid import Find_Centroid 
from Function_Intensity_Image import Intensity_Image 
from Function_Decompose_Spectrum import Decompose_Spectrum 
from Function_Phase_Decomposition import Phase_Decomposition 
from Function_Gerchberg_Saxton_Algorithm import Gerchberg_Saxton_Algorithm 

# Phase Unwrapping 
from skimage.restoration import unwrap_phase 


###############################################################################
# Intensity Data 
###############################################################################

start = time.time() 
########## ----- Read h5 file Data ----- ########## 
name    = 'longitudinal focal shift/test_SO_laser_lentille_4' 
file    = f'C:/Users/APPLI/Desktop/Haoxuan Python/Experimental Data/{name}.h5' 
h5      = h5py.File(file, 'r') 
data    = h5['data'] 
energy  = h5['energy'] 
scales  = h5['scales'] 
density = h5['filters'] 
del(h5) 

t  = np.asarray(scales['tau']) * 1e-15   # The t-axis in $s$. 
x  = np.asarray(scales['xy']) * 1e-3     # The xy-axis in $m$. 
l  = 532 * 1e-9                          # The wavelength in $m$. 
zR = 6.025 * 1e-3                        # The Rayleigh range in $m$. 
D  = 12.05 * 1e-3                        # The distance between planes in $m$. 

########## ----- Get Intensity Data in Each Plane ----- ########## 
data_n = Get_Data(data, 'n') 
data_0 = Get_Data(data, '0') 
data_p = Get_Data(data, 'p') 

########## ----- Plot "Get Data" in Each Plane ----- ########## 
fg1 = plt.figure(1, figsize=(20,16)) 
fg1.suptitle('Figure 1 : The Intensity Data in Each Plane', fontsize=24) 

ax1 = fg1.add_subplot(311) 
ax1.plot(t, data_n) 
ax1.set_xlabel('Tau ($s$)', fontsize=12) 
ax1.set_ylabel('Intensity in Plane z=n', fontsize=12) 
ax1.grid() 

ax2 = fg1.add_subplot(312) 
ax2.plot(t, data_0) 
ax2.set_xlabel('Tau ($s$)', fontsize=12) 
ax2.set_ylabel('Intensity in Plane z=0', fontsize=12) 
ax2.grid() 

ax3 = fg1.add_subplot(313) 
ax3.plot(t, data_p) 
ax3.set_xlabel('Tau ($s$)', fontsize=12) 
ax3.set_ylabel('Intensity in Plane z=p', fontsize=12) 
ax3.grid() 

########## ----- Get Spectrum for Each Pixel & Each Plane ----- ########## 
fn, An, sum_An = Get_Spectrum(data, x, t, 'n', 3) 
f0, A0, sum_A0 = Get_Spectrum(data, x, t, '0', 3) 
fp, Ap, sum_Ap = Get_Spectrum(data, x, t, 'p', 3) 

left  = 600 * 1e-9   # The left bound wavelength in $m$. 
right = 950 * 1e-9   # The right bound wavelength in $m$. 

fn, ln, An, sum_An, sum_An_l = Convert_Spectrum(fn, An, sum_An, left, right) 
f0, l0, A0, sum_A0, sum_A0_l = Convert_Spectrum(f0, A0, sum_A0, left, right) 
fp, lp, Ap, sum_Ap, sum_Ap_l = Convert_Spectrum(fp, Ap, sum_Ap, left, right) 

########## ----- Plot "Get Spectrum" in Each Plane ----- ########## 
fg2 = plt.figure(2, figsize=(20,10)) 
fg2.suptitle('Figure 2 : The Wavelength Spectrum in Each Plane', fontsize=24) 

ax1 = fg2.add_subplot(131) 
ax1.plot(ln, sum_An_l, marker="o", label='dI/dlambda') 
ax1.plot(ln, sum_An, marker="o", label='dI/domega')
ax1.set_title('Plane z=n', fontsize=12) 
ax1.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax1.set_ylabel('Intensity', fontsize=12) 
ax1.legend() 
ax1.grid() 

ax2 = fg2.add_subplot(132) 
ax2.plot(l0, sum_A0_l, marker="o", label='dI/dlambda') 
ax2.plot(l0, sum_A0, marker="o", label='dI/domega')
ax2.set_title('Plane z=0', fontsize=12) 
ax2.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_ylabel('Intensity', fontsize=12) 
ax2.legend() 
ax2.grid() 

ax3 = fg2.add_subplot(133) 
ax3.plot(lp, sum_Ap_l, marker="o", label='dI/dlambda') 
ax3.plot(lp, sum_Ap, marker="o", label='dI/domega')
ax3.set_title('Plane z=p', fontsize=12) 
ax3.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax3.set_ylabel('Intensity', fontsize=12) 
ax3.legend() 
ax3.grid() 

end = time.time() 
print('"Intensity Data" is Done !') 
print(f'The execution time is {end - start} seconds.') 


###############################################################################
# Gerchberg-Saxton Algorithm Preparation 
###############################################################################

start_2 = time.time() 
########## ----- Useful Parameters ----- ########## 
k      = (2*np.pi) / (l*1e-6)         # The spatial frequency in unit $mm^{-1}$. 
w0     = np.sqrt(zR*l*1e-6 / np.pi)   # The theoretical beam width in unit $mm$.  
dx     = x[1] - x[0]                  # The size of one pixel in unit $mm$. 
N      = np.size(x)                   # The number of pixels. 
F      = 1/dx 
df     = F/N 
kx     = np.linspace(-(N/2)*df*2*np.pi, (N/2 - 1)*df*2*np.pi, N) 
kX, kY = np.meshgrid(kx, kx) 
kZ     = np.real(np.emath.sqrt(k**2 - kX**2 - kY**2)) 

########## ----- Initial Estimation of Phase ----- ########## 
X, Y   = np.meshgrid(x, x) 
radius = np.sqrt(X**2 + Y**2) / (2*w0) 
theta  = np.arctan2(Y, X) 
phase0 = Phase_Generator(radius, theta, 0, 1, 1, 0, 0) 

iteration = 20 

end_2 = time.time() 
print('"Gerchberg-Saxton Algorithm Preparation" is Done !') 
print(f'The execution time is {end_2 - start_2} seconds.') 


###############################################################################
# Spectrum Analysis -- Axis Analysis 
###############################################################################

start_3 = time.time() 
########## ----- Axis Analysis in z=0 ----- ########## 
Intensity_all = np.average(A0, axis=2) 
X_Spectrum    = np.transpose(np.sum(A0, axis=1)) 
Y_Spectrum    = np.transpose(np.sum(A0, axis=0)) 

Wavelength_AA = [left+10, int((left+right)/2), right-10]   # Drawing circles. 

Center_x_AA = [] 
Center_y_AA = [] 
Sigma_x_AA  = [] 
Sigma_y_AA  = [] 

for wavelength in Wavelength_AA: 
    Intensity = Intensity_Image(l0, A0, wavelength) 
    Intensity = Threshold(Intensity, 0.1) 
    
    x0, y0 = Find_Centroid(x, Intensity) 
    Center_x_AA.append(x0) 
    Center_y_AA.append(y0) 
    
    sigma_x, sigma_y = Find_Sigma(x, Intensity) 
    Sigma_x_AA.append(sigma_x) 
    Sigma_y_AA.append(sigma_y) 

########## ----- Plot "Axis Analysis" in z=0 ----- ########## 
fg3 = plt.figure(3, figsize=(25,6)) 
fg3.suptitle('Figure 3 : The Axis Decomposition in z=0', fontsize=24) 

color_list = ['blue','green','red'] 

ax1 = fg3.add_subplot(131) 
im1 = ax1.pcolormesh(x, x, Intensity_all / np.max(Intensity_all), cmap=plt.cm.jet, shading='auto') 
ax1.set_title('Intensity at plane z=0', fontsize=12) 
ax1.set_xlabel('Spatial Distance $x$ ($mm$)', fontsize=12) 
ax1.set_ylabel('Spatial Distance $y$ ($mm$)', fontsize=12) 
for i in range(len(Wavelength_AA)): 
    R = Sigma_x_AA[i] + Sigma_y_AA[i] 
    C = Circle((Center_x_AA[i], Center_y_AA[i]), R, linestyle='--', facecolor='None', 
               edgecolor=color_list[i], lw=2, zorder=10, label=f'{Wavelength_AA[i]} $nm$') 
    ax1.add_patch(C) 
ax1.legend() 
plt.colorbar(im1, ax=ax1) 

ax2 = fg3.add_subplot(132) 
im2 = ax2.pcolormesh(x, l0, X_Spectrum / np.max(X_Spectrum), cmap=plt.cm.jet, shading='auto') 
ax2.set_title('x-Spectrum at plane z=0', fontsize=12) 
ax2.set_xlabel('Spatial Distance $x$ ($mm$)', fontsize=12) 
ax2.set_ylabel('Wavelength ($nm$)', fontsize=12) 
plt.colorbar(im2, ax=ax2) 

ax3 = fg3.add_subplot(133) 
im3 = ax3.pcolormesh(x, l0, Y_Spectrum / np.max(Y_Spectrum), cmap=plt.cm.jet, shading='auto') 
ax3.set_title('y-Spectrum at plane z=0', fontsize=12) 
ax3.set_xlabel('Spatial Distance $y$ ($mm$)', fontsize=12) 
ax3.set_ylabel('Wavelength ($nm$)', fontsize=12) 
plt.colorbar(im3, ax=ax3) 

end_3 = time.time() 
print('"Axis Decomposition" is Done !') 
print(f'The execution time is {end_3 - start_3} seconds.') 


###############################################################################
# Spectrum Analysis -- Wavelength Analysis 
###############################################################################

start_4 = time.time() 
########## ----- Wavelength Analysis in z=0 ----- ########## 
Wavelength = 700 
if (Wavelength < left + 10) or (Wavelength > right - 10): 
    raise ValueError 

Intensity_n = Intensity_Image(ln, An, Wavelength) 
Intensity_0 = Intensity_Image(l0, A0, Wavelength) 
Intensity_p = Intensity_Image(lp, Ap, Wavelength) 

Phase, I0_updated = Gerchberg_Saxton_Algorithm(Intensity_n, Intensity_0, Intensity_p, 
                                               phase0, kZ, D, iteration) 
Phase[radius > 1] = 0 
Phase = unwrap_phase(Phase) 


########## ----- Plot "Wavelength Analysis" in z=0 ----- ########## 
fg4 = plt.figure(4, figsize=(25,6)) 
fg4.suptitle('Figure 4 : The Wavelength Analysis in z=0', fontsize=24) 

ax1 = fg4.add_subplot(131) 
im1 = ax1.pcolormesh(x, x, Intensity_0 / np.max(Intensity_0), cmap=plt.cm.jet, shading='auto') 
ax1.set_title(f'Measured Intensity of {Wavelength} $\pm 5 nm$', fontsize=12) 
ax1.set_xlabel('Spatial Distance $x$ ($mm$)', fontsize=12) 
ax1.set_ylabel('Spatial Distance $y$ ($mm$)', fontsize=12) 
plt.colorbar(im1, ax=ax1) 

ax2 = fg4.add_subplot(132) 
im2 = ax2.pcolormesh(x, x, I0_updated / np.max(I0_updated), cmap=plt.cm.jet, shading='auto') 
ax2.set_title(f'Reconstructed Intensity of {Wavelength} $\pm 5 nm$', fontsize=12) 
ax2.set_xlabel('Spatial Distance $x$ ($mm$)', fontsize=12) 
ax2.set_ylabel('Spatial Distance $y$ ($mm$)', fontsize=12) 
plt.colorbar(im2, ax=ax2) 

ax3 = fg4.add_subplot(133) 
im3 = ax3.pcolormesh(x, x, Phase, cmap=plt.cm.jet, shading='auto') 
ax3.set_title('Reconstructed Phase', fontsize=12) 
ax3.set_xlabel('Spatial Distance $x$ ($mm$)', fontsize=12) 
ax3.set_ylabel('Spatial Distance $y$ ($mm$)', fontsize=12) 
plt.colorbar(im3, ax=ax3) 

end_4 = time.time() 
print('"Wavelength Analysis" is Done !') 
print(f'The execution time is {end_4 - start_4} seconds.') 


###############################################################################
# Spectrum Analysis -- Transverse Focal Shift 
###############################################################################

start_5 = time.time() 
########## ----- Transverse Focal Shift in z=0 ----- ########## 
Wavelength_TFS = list(np.linspace(left+10, right-10, 51)) 
Center_x_TFS   = [] 
Center_y_TFS   = [] 
Sigma_x_TFS    = [] 
Sigma_y_TFS    = [] 

for wavelength in Wavelength_TFS: 
    Intensity = Intensity_Image(l0, A0, wavelength) 
    Intensity = Threshold(Intensity, 0.1) 
    
    x0, y0 = Find_Centroid(x, Intensity) 
    Center_x_TFS.append(x0) 
    Center_y_TFS.append(y0) 
    
    sigma_x, sigma_y = Find_Sigma(x, Intensity) 
    Sigma_x_TFS.append(sigma_x) 
    Sigma_y_TFS.append(sigma_y) 

########## ----- Plot "Transverse Focal Shift" in z=0 ----- ########## 
fg5 = plt.figure(5, figsize=(25, 15)) 
fg5.suptitle('Figure 5 : The Transverse Focal Shift in z=0', fontsize=24) 

ax1 = fg5.add_subplot(221) 
ax1.plot(Wavelength_TFS, np.array(Center_x_TFS)*1e3, marker="o") 
ax1.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax1.set_ylabel('Centroid-x ($\mu m$)', fontsize=12) 
ax1.grid() 

ax2 = fg5.add_subplot(222) 
ax2.plot(Wavelength_TFS, np.array(Center_y_TFS)*1e3, marker="o") 
ax2.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_ylabel('Centroid-y ($\mu m$)', fontsize=12) 
ax2.grid() 

ax3 = fg5.add_subplot(223) 
ax3.plot(Wavelength_TFS, np.array(Sigma_x_TFS)*1e3, marker="o") 
ax3.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax3.set_ylabel('Sigma-x ($\mu m$)', fontsize=12) 
ax3.grid() 

ax4 = fg5.add_subplot(224) 
ax4.plot(Wavelength_TFS, np.array(Sigma_y_TFS)*1e3, marker="o") 
ax4.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax4.set_ylabel('Sigma-y ($\mu m$)', fontsize=12) 
ax4.grid() 

end_5 = time.time() 
print('"Transverse Focal Shift" is Done !') 
print(f'The execution time is {end_5 - start_5} seconds.') 


###############################################################################
# Spectrum Analysis -- Longitudinal Focal Shift (Method 1) 
###############################################################################

start_6 = time.time() 
########## ----- Longitudinal Focal Shift in z ----- ########## 
Wavelength_LFS = [650, 750, 850] 
if (len(Wavelength_LFS) > 3): 
    raise ValueError 
for wavelength in Wavelength_LFS: 
    if (wavelength < left + 10) or (wavelength > right - 10): 
        raise ValueError 

Intensity_An_LFS = Decompose_Spectrum(ln, An, Wavelength_LFS) 
Intensity_A0_LFS = Decompose_Spectrum(l0, A0, Wavelength_LFS) 
Intensity_Ap_LFS = Decompose_Spectrum(lp, Ap, Wavelength_LFS) 

Phase_LFS = [] 
for i in range(len(Wavelength_LFS)): 
    intensity_n = Intensity_An_LFS[i] 
    intensity_0 = Intensity_A0_LFS[i] 
    intensity_p = Intensity_Ap_LFS[i] 
    
    phase, I0_updated = Gerchberg_Saxton_Algorithm(intensity_n, intensity_0, intensity_p, 
                                                   phase0, kZ, D, iteration) 
    Phase_LFS.append(phase) 

Distance_LFS = list(np.linspace(-5, 5, 101)) 
Sigma_x_LFS  = []   # The sigma_x of all wavelength. 
Sigma_y_LFS  = []   # The sigma_y of all wavelength. 
Shift_x      = []   # The position of minimum sigma_x of all wavelength. 
Shift_y      = []   # The position of minimum sigma_y of all wavelength. 

for i in range(len(Wavelength_LFS)): 
    Intensity = Intensity_A0_LFS[i] 
    phase     = Phase_LFS[i] 
    E1        = np.sqrt(Intensity) * np.exp(1j*phase) 
    
    sigma_x_w = []   # The sigma_x of all distance for this wavelength. 
    sigma_y_w = []   # The sigma_y of all distance for this wavelength. 
    
    for distance in Distance_LFS: 
        Eout = Propagator(E1, kZ, distance) 
        Iout = np.real(Eout * np.conj(Eout)) 
        Iout = Threshold(Iout, 0.1) 
        
        sigma_x, sigma_y = Find_Sigma(x, Iout) 
        sigma_x_w.append(sigma_x) 
        sigma_y_w.append(sigma_y) 
    
    Sigma_x_LFS.append(sigma_x_w) 
    Sigma_y_LFS.append(sigma_y_w) 
    
    index_min_x = np.where(sigma_x_w == np.min(sigma_x_w))[0][0] 
    index_min_y = np.where(sigma_y_w == np.min(sigma_y_w))[0][0] 
    Shift_x.append(Distance_LFS[index_min_x]) 
    Shift_y.append(Distance_LFS[index_min_y]) 

########## ----- Plot "Longitudinal Focal Shift (M1)" in z ----- ########## 
color_list = ['blue','green','red'] 

fg6 = plt.figure(6, figsize=(25, 15)) 
fg6.suptitle('Figure 6 : The Longitudinal Focal Shift in z', fontsize=24) 

ax1 = fg6.add_subplot(221) 
for i in range(len(Wavelength_LFS)): 
    ax1.plot(Distance_LFS, np.array(Sigma_x_LFS[i])*1e3, color=color_list[i], 
             label=f'{Wavelength_LFS[i]} nm') 
ax1.set_xlabel('Spatial Distance $z$ ($mm$)', fontsize=12) 
ax1.set_ylabel('Sigma-x ($\mu m$)', fontsize=12) 
ax1.legend() 
ax1.grid() 

ax2 = fg6.add_subplot(222) 
for i in range(len(Wavelength_LFS)): 
    ax2.plot(Distance_LFS, np.array(Sigma_y_LFS[i])*1e3, color=color_list[i], 
             label=f'{Wavelength_LFS[i]} nm') 
ax2.set_xlabel('Spatial Distance $z$ ($mm$)', fontsize=12) 
ax2.set_ylabel('Sigma-y ($\mu m$)', fontsize=12) 
ax2.legend() 
ax2.grid() 

ax3 = fg6.add_subplot(223) 
ax3.plot(Wavelength_LFS, Shift_x, marker="o") 
ax3.set_title('Minimum Shift in X') 
ax3.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax3.set_ylabel('Spatial Distance $z$ ($mm$)', fontsize=12) 
ax3.grid() 

ax4 = fg6.add_subplot(224) 
ax4.plot(Wavelength_LFS, Shift_y, marker="o") 
ax4.set_title('Minimum Shift in Y') 
ax4.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax4.set_ylabel('Spatial Distance $z$ ($mm$)', fontsize=12) 
ax4.grid() 

end_6 = time.time() 
print('"Longitudinal Focal Shift" is Done !') 
print(f'The execution time is {end_6 - start_6} seconds.') 


###############################################################################
# Spectrum Analysis -- Longitudinal Focal Shift (Method 2) 
###############################################################################

start_6 = time.time() 
########## ----- Longitudinal Focal Shift in z (M2) ----- ########## 
Wavelength_LFS = [650, 750, 850] 
if (len(Wavelength_LFS) > 3): 
    raise ValueError 
for wavelength in Wavelength_LFS: 
    if (wavelength < left + 10) or (wavelength > right - 10): 
        raise ValueError 

Intensity_An_LFS = Decompose_Spectrum(ln, An, Wavelength_LFS) 
Intensity_A0_LFS = Decompose_Spectrum(l0, A0, Wavelength_LFS) 
Intensity_Ap_LFS = Decompose_Spectrum(lp, Ap, Wavelength_LFS) 

Phase_LFS = [] 
for i in range(len(Wavelength_LFS)): 
    intensity_n = Intensity_An_LFS[i] 
    intensity_0 = Intensity_A0_LFS[i] 
    intensity_p = Intensity_Ap_LFS[i] 
    
    phase, I0_updated = Gerchberg_Saxton_Algorithm(intensity_n, intensity_0, intensity_p, 
                                                   phase0, kZ, D, iteration) 
    Phase_LFS.append(phase) 

Distance_LFS  = list(np.linspace(-10, 10, 201)) 
Intensity_LFS = [[], [], []] 
Shift_LFS     = [] 
for i in range(len(Wavelength_LFS)): 
    Intensity = Intensity_A0_LFS[i] 
    phase     = Phase_LFS[i] 
    E1        = np.sqrt(Intensity) * np.exp(1j*phase) 
    
    for distance in Distance_LFS: 
        Eout = Propagator(E1, kZ, distance) 
        Iout = np.real(Eout * np.conj(Eout)) 
        Iout = Iout.flatten() 
        Iout.sort() 
        Intensity_LFS[i].append(Iout[-1] + Iout[-2] + Iout[-3] + Iout[-4]) 
    
    index_max = np.where(Intensity_LFS[i] == np.max(Intensity_LFS[i]))[0][0] 
    Shift_LFS.append(Distance_LFS[index_max]) 

########## ----- Plot "Longitudinal Focal Shift (M2)" in z ----- ########## 
color_list = ['blue','green','red'] 

fg10 = plt.figure(10, figsize=(25, 15)) 
fg10.suptitle('Figure 10 : The Longitudinal Focal Shift in z', fontsize=24) 

ax1 = fg10.add_subplot(211) 
for i in range(len(Wavelength_LFS)): 
    ax1.plot(Distance_LFS, np.array(Intensity_LFS[i]), color=color_list[i], 
             label=f'{Wavelength_LFS[i]} nm') 
ax1.set_xlabel('Spatial Distance $z$ ($mm$)', fontsize=12) 
ax1.set_ylabel('Intensity', fontsize=12) 
ax1.legend() 
ax1.grid() 

ax2 = fg10.add_subplot(212) 
ax2.plot(Wavelength_LFS, Shift_LFS, marker="o") 
ax2.set_title('Minimum Shift') 
ax2.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_ylabel('Spatial Distance $z$ ($mm$)', fontsize=12) 
ax2.grid() 

end_6 = time.time() 
print('"Longitudinal Focal Shift" is Done !') 
print(f'The execution time is {end_6 - start_6} seconds.') 


###############################################################################
# Spectrum Analysis -- Phase Decomposition in Far Field 
###############################################################################

start_7 = time.time() 
########## ----- Phase Decomposition in Far Field ----- ########## 
Wavelength_PD = list(np.linspace(left+10, right-10, int(((right-10)-(left+10))/10 + 1))) 

Intensity_An_PD = Decompose_Spectrum(ln, An, Wavelength_PD) 
Intensity_A0_PD = Decompose_Spectrum(l0, A0, Wavelength_PD) 
Intensity_Ap_PD = Decompose_Spectrum(lp, Ap, Wavelength_PD) 

Phase_FF = [] 
for i in range(len(Wavelength_PD)): 
    intensity_n = Intensity_An_PD[i] 
    intensity_0 = Intensity_A0_PD[i] 
    intensity_p = Intensity_Ap_PD[i] 
    
    phase, I0_updated = Gerchberg_Saxton_Algorithm(intensity_n, intensity_0, intensity_p, 
                                                   phase0, kZ, D, iteration) 
    Phase_FF.append(phase) 

N_aberration    = 8   # There are 8 kinds of aberrations. 
Coefficient_FF  = [] 
Phase_FF_unwrap = np.copy(Phase_FF) 
for i in range(N_aberration): 
    Coefficient_FF.append([]) 
for phase in Phase_FF_unwrap: 
    phase[radius > 1] = 0 
    phase = unwrap_phase(phase) 
    coefficient = Phase_Decomposition(radius, theta, phase) 
    for i in range(N_aberration): 
        Coefficient_FF[i].append(coefficient[i]) 

########## ----- Plot "Phase Decomposition in Far Field" ----- ########## 
fg7 = plt.figure(7, figsize=(30, 15)) 
fg7.suptitle('Figure 7 : The Phase Decomposition in Far Field', fontsize=24) 

ax1 = fg7.add_subplot(241) 
ax1.plot(Wavelength_PD, Coefficient_FF[0], marker="o") 
ax1.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax1.set_ylabel('Tilt Vertical', fontsize=12) 
ax1.grid() 

ax2 = fg7.add_subplot(242) 
ax2.plot(Wavelength_PD, Coefficient_FF[1], marker="o") 
ax2.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_ylabel('Tilt Horizontal', fontsize=12) 
ax2.grid() 

ax3 = fg7.add_subplot(243) 
ax3.plot(Wavelength_PD, Coefficient_FF[2], marker="o") 
ax3.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax3.set_ylabel('Astigmatism Oblique', fontsize=12) 
ax3.grid() 

ax4 = fg7.add_subplot(244) 
ax4.plot(Wavelength_PD, Coefficient_FF[3], marker="o") 
ax4.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax4.set_ylabel('Astigmatism Horizontal', fontsize=12) 
ax4.grid() 

ax5 = fg7.add_subplot(245) 
ax5.plot(Wavelength_PD, Coefficient_FF[4], marker="o") 
ax5.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax5.set_ylabel('Defocus', fontsize=12) 
ax5.grid() 

ax6 = fg7.add_subplot(246) 
ax6.plot(Wavelength_PD, Coefficient_FF[5], marker="o") 
ax6.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax6.set_ylabel('Coma Vertical', fontsize=12) 
ax6.grid() 

ax7 = fg7.add_subplot(247) 
ax7.plot(Wavelength_PD, Coefficient_FF[6], marker="o") 
ax7.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax7.set_ylabel('Coma Horizontal', fontsize=12) 
ax7.grid() 

ax8 = fg7.add_subplot(248) 
ax8.plot(Wavelength_PD, Coefficient_FF[7], marker="o") 
ax8.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax8.set_ylabel('Spherical', fontsize=12) 
ax8.grid() 

end_7 = time.time() 
print('"Phase Decomposition in Far Field" is Done !') 
print(f'The execution time is {end_7 - start_7} seconds.') 


###############################################################################
# Spectrum Analysis -- Near Field Analysis 
###############################################################################

start_8 = time.time() 
########## ----- Near Field Analysis ----- ########## 
w1      = 20                                          # The size of Gaussian beam in unit $mm$. 
S1      = 200                                         # The size of object plane in unit $mm$. 
ds1     = S1/N                                        # The size of one pixel. 
x1      = np.linspace(-(N/2)*ds1, (N/2 - 1)*ds1, N)   # The axis in object plane. 
X1, Y1  = np.meshgrid(x1, x1)                         # The xy-grid in object plane. 
radius1 = np.sqrt(X1**2 + Y1**2) / (2*w1) 
theta1  = np.arctan2(Y1, X1) 

E0_all = 0    # The total wave in the near field. 
E0_idv = []   # The wave in the near field for different wavelength. 
for i in range(len(Wavelength_PD)): 
    E1 = np.sqrt(Intensity_A0_PD[i]) * np.exp(1j*Phase_FF[i]) 
    E0 = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(E1))) 
    E0_idv.append(E0) 
    E0_all += E0 

Intensity_NF = np.real(E0_all * np.conj(E0_all)) 
Phase_NF = np.angle(E0_all) 
Phase_NF[radius1 > 1] = 0 
Phase_NF = unwrap_phase(Phase_NF) 

########## ----- Plot "Near Field Analysis" ----- ########## 
fg8 = plt.figure(8, figsize=(25, 10)) 
fg8.suptitle('Figure 8 : The Near Field Analysis', fontsize=24) 

ax1 = fg8.add_subplot(121) 
im1 = ax1.pcolormesh(x1, x1, Intensity_NF, cmap=plt.cm.jet, shading='auto') 
ax1.set_title('Reconstructed Intensity in Near Field', fontsize=12) 
ax1.set_xlabel('Spatial Distance $x$ ($mm$)', fontsize=12) 
ax1.set_ylabel('Spatial Distance $y$ ($mm$)', fontsize=12) 
#ax1.set_xlim([-2.5, 2.5]) 
#ax1.set_ylim([-2.5, 2.5]) 
plt.colorbar(im1, ax=ax1) 

ax2 = fg8.add_subplot(122) 
im2 = ax2.pcolormesh(x1, x1, Phase_NF, cmap=plt.cm.jet, shading='auto') 
ax2.set_title('Reconstructed Phase in Near Field', fontsize=12) 
ax2.set_xlabel('Spatial Distance $x$ ($mm$)', fontsize=12) 
ax2.set_ylabel('Spatial Distance $y$ ($mm$)', fontsize=12) 
#ax2.set_xlim([-2.5, 2.5]) 
#ax2.set_ylim([-2.5, 2.5]) 
plt.colorbar(im2, ax=ax2) 

end_8 = time.time() 
print('"Near Field Analysis" is Done !') 
print(f'The execution time is {end_8 - start_8} seconds.') 


###############################################################################
# Spectrum Analysis -- Phase Decomposition in Near Field 
###############################################################################

start_9 = time.time() 
########## ----- Phase Decomposition in Near Field ----- ########## 
Coefficient_NF = [] 
for i in range(N_aberration): 
    Coefficient_NF.append([]) 
for wave in E0_idv: 
    phase = np.angle(wave) 
    phase[radius1 > 1] = 0 
    phase = unwrap_phase(phase) 
    coefficient = Phase_Decomposition(radius1, theta1, phase) 
    for i in range(N_aberration): 
        Coefficient_NF[i].append(coefficient[i]) 

########## ----- Plot "Phase Decomposition in Near Field" ----- ########## 
fg9 = plt.figure(9, figsize=(25, 15)) 
fg9.suptitle('Figure 9 : The Phase Decomposition in Near Field', fontsize=24) 

ax1 = fg9.add_subplot(241) 
ax1.plot(Wavelength_PD, Coefficient_NF[0], marker="o") 
ax1.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax1.set_ylabel('Tilt Vertical', fontsize=12) 
ax1.grid() 

ax2 = fg9.add_subplot(242) 
ax2.plot(Wavelength_PD, Coefficient_NF[1], marker="o") 
ax2.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_ylabel('Tilt Horizontal', fontsize=12) 
ax2.grid() 

ax3 = fg9.add_subplot(243) 
ax3.plot(Wavelength_PD, Coefficient_NF[2], marker="o") 
ax3.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax3.set_ylabel('Astigmatism Oblique', fontsize=12) 
ax3.grid() 

ax4 = fg9.add_subplot(244) 
ax4.plot(Wavelength_PD, Coefficient_NF[3], marker="o") 
ax4.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax4.set_ylabel('Astigmatism Horizontal', fontsize=12) 
ax4.grid() 

ax5 = fg9.add_subplot(245) 
ax5.plot(Wavelength_PD, Coefficient_NF[4], marker="o") 
ax5.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax5.set_ylabel('Defocus', fontsize=12) 
ax5.grid() 

ax6 = fg9.add_subplot(246) 
ax6.plot(Wavelength_PD, Coefficient_NF[5], marker="o") 
ax6.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax6.set_ylabel('Coma Vertical', fontsize=12) 
ax6.grid() 

ax7 = fg9.add_subplot(247) 
ax7.plot(Wavelength_PD, Coefficient_NF[6], marker="o") 
ax7.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax7.set_ylabel('Coma Horizontal', fontsize=12) 
ax7.grid() 

ax8 = fg9.add_subplot(248) 
ax8.plot(Wavelength_PD, Coefficient_NF[7], marker="o") 
ax8.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax8.set_ylabel('Spherical', fontsize=12) 
ax8.grid() 

end_9 = time.time() 
print('"Phase Decomposition in Near Field" is Done !') 
print(f'The execution time is {end_9 - start_9} seconds.') 


###############################################################################
# Total Execution Time 
###############################################################################

