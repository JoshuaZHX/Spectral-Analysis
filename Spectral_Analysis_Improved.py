""" 
Program : Spectral_Analysis_Improved 
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

# Spectral Analysis 
from Function_Phase_Cut import Phase_Cut 
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
file    = f'D:/Internships/Internship -- LOA 2022/Experimental Data/{name}.h5' 
h5      = h5py.File(file, 'r') 
data    = h5['data'] 
energy  = h5['energy'] 
scales  = h5['scales'] 
density = h5['filters'] 
del(h5) 

t  = np.asarray(scales['tau']) * 1e-15   # The t-axis in $s$. 
x  = np.asarray(scales['xy']) * 1e-3     # The xy-axis in $m$. 
zR = 0.1 * 1e-3   # The Rayleigh range in $m$. 
D  = 0.3 * 1e-3   # The distance between planes in $m$. 

########## ----- Get Intensity Data in Each Plane ----- ########## 
data_n = Get_Data(data, 'n') 
data_0 = Get_Data(data, '0') 
data_p = Get_Data(data, 'p') 

########## ----- Plot "Get Data" in Each Plane ----- ########## 
fg1 = plt.figure(1, figsize=(20,16)) 
fg1.suptitle('Figure 1 : The Intensity Data in Each Plane', fontsize=24) 

ax1 = fg1.add_subplot(311) 
ax1.plot(t*1e15, data_n) 
ax1.set_xlabel('Tau ($fs$)', fontsize=12) 
ax1.set_ylabel('Intensity in Plane z=n', fontsize=12) 
ax1.grid() 

ax2 = fg1.add_subplot(312) 
ax2.plot(t*1e15, data_0) 
ax2.set_xlabel('Tau ($fs$)', fontsize=12) 
ax2.set_ylabel('Intensity in Plane z=0', fontsize=12) 
ax2.grid() 

ax3 = fg1.add_subplot(313) 
ax3.plot(t*1e15, data_p) 
ax3.set_xlabel('Tau ($fs$)', fontsize=12) 
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
ax1.plot(ln*1e9, sum_An_l, marker="o", label='dI/dlambda') 
ax1.plot(ln*1e9, sum_An, marker="o", label='dI/domega')
ax1.set_title('Plane z=n', fontsize=12) 
ax1.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax1.set_ylabel('Intensity', fontsize=12) 
ax1.legend() 
ax1.grid() 

ax2 = fg2.add_subplot(132) 
ax2.plot(l0*1e9, sum_A0_l, marker="o", label='dI/dlambda') 
ax2.plot(l0*1e9, sum_A0, marker="o", label='dI/domega')
ax2.set_title('Plane z=0', fontsize=12) 
ax2.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_ylabel('Intensity', fontsize=12) 
ax2.legend() 
ax2.grid() 

ax3 = fg2.add_subplot(133) 
ax3.plot(lp*1e9, sum_Ap_l, marker="o", label='dI/dlambda') 
ax3.plot(lp*1e9, sum_Ap, marker="o", label='dI/domega')
ax3.set_title('Plane z=p', fontsize=12) 
ax3.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax3.set_ylabel('Intensity', fontsize=12) 
ax3.legend() 
ax3.grid() 

end = time.time() 
print('"Intensity Data" is Done !') 
print(f'The execution time is {end - start} seconds.') 


###############################################################################
# Spectrum Analysis -- Axis Analysis 
###############################################################################

start = time.time() 
########## ----- Axis Analysis in z=0 ----- ########## 
Intensity_all = np.average(A0, axis=2) 
X_Spectrum    = np.transpose(np.sum(A0, axis=1)) 
Y_Spectrum    = np.transpose(np.sum(A0, axis=0)) 

Wavelength_AA = [left+(10*1e-9), (left+right)/2, right-(10*1e-9)] 

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

center_x = Center_x_AA[1] 
center_y = Center_y_AA[1] 
sigma    = np.max([Sigma_x_AA[1], Sigma_y_AA[1]]) 

########## ----- Plot "Axis Analysis" in z=0 ----- ########## 
fg3 = plt.figure(3, figsize=(25,6)) 
fg3.suptitle('Figure 3 : The Axis Decomposition in z=0', fontsize=24) 

color_list = ['white','orange','red'] 

ax1 = fg3.add_subplot(131) 
im1 = ax1.pcolormesh(x, x, Intensity_all / np.max(Intensity_all), cmap=plt.cm.jet, shading='auto') 
ax1.set_title('Intensity at plane z=0', fontsize=12) 
ax1.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax1.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
ax1.set_xlim([center_x - 8*sigma, center_x + 8*sigma]) 
ax1.set_ylim([center_y - 8*sigma, center_y + 8*sigma]) 
for i in range(len(Wavelength_AA)): 
    R = Sigma_x_AA[i] + Sigma_y_AA[i] 
    C = Circle((Center_x_AA[i], Center_y_AA[i]), R, linestyle='--', facecolor='None', 
               edgecolor=color_list[i], lw=2, zorder=10, label=f'{Wavelength_AA[i]} $m$') 
    ax1.add_patch(C) 
ax1.legend() 
plt.colorbar(im1, ax=ax1) 

ax2 = fg3.add_subplot(132) 
im2 = ax2.pcolormesh(x, l0*1e9, X_Spectrum / np.max(X_Spectrum), cmap=plt.cm.jet, shading='auto') 
ax2.set_title('x-Spectrum at plane z=0', fontsize=12) 
ax2.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax2.set_ylabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_xlim([center_y - 8*sigma, center_y + 8*sigma]) 
plt.colorbar(im2, ax=ax2) 

ax3 = fg3.add_subplot(133) 
im3 = ax3.pcolormesh(x, l0*1e9, Y_Spectrum / np.max(Y_Spectrum), cmap=plt.cm.jet, shading='auto') 
ax3.set_title('y-Spectrum at plane z=0', fontsize=12) 
ax3.set_xlabel('Spatial Distance $y$ ($m$)', fontsize=12) 
ax3.set_ylabel('Wavelength ($nm$)', fontsize=12) 
ax3.set_xlim([center_x - 8*sigma, center_x + 8*sigma]) 
plt.colorbar(im3, ax=ax3) 

end = time.time() 
print('"Axis Analysis" is Done !') 
print(f'The execution time is {end - start} seconds.') 


###############################################################################
# Spectrum Analysis -- Wavelength Analysis 
###############################################################################

start = time.time() 
########## ----- Wavelength Analysis in z=0 ----- ########## 
Wavelength = 800 * 1e-9 
if (Wavelength < left + (10*1e-9)) or (Wavelength > right - (10*1e-9)): 
    raise ValueError 

intensity_n = Intensity_Image(ln, An, Wavelength) 
intensity_0 = Intensity_Image(l0, A0, Wavelength) 
intensity_p = Intensity_Image(lp, Ap, Wavelength) 

iteration  = 50                # The number of iterations for GS algorithm. 
estimation = [0, 0, 0, 0, 0]   # The initial phase estimation coefficients. 
Phase, I0_updated, In_updated, Ip_updated, Error1, Error2, Error3 = Gerchberg_Saxton_Algorithm(intensity_n, intensity_0, intensity_p, Wavelength, x, zR, estimation, D, iteration) 

intensity = np.copy(intensity_0) 
intensity = Threshold(intensity, 0.1) 
x0, y0    = Find_Centroid(x, Intensity) 
sigma     = np.max([Find_Sigma(x, intensity)]) 

Phase = Phase_Cut(Phase, x, x0, y0, sigma) 
Phase = unwrap_phase(Phase) 

########## ----- Plot "Wavelength Analysis" in z=0 ----- ########## 
fg4 = plt.figure(4, figsize=(30,23)) 
fg4.suptitle('Figure 4 : The Wavelength Analysis in z=0', fontsize=24) 

ax1 = fg4.add_subplot(331) 
im1 = ax1.pcolormesh(x, x, intensity_n / np.max(intensity_n), cmap=plt.cm.jet, shading='auto') 
ax1.set_title(f'Measured Intensity of {int(Wavelength * 1e9)} $\pm 5 nm$ in Plane n', fontsize=12) 
ax1.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax1.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
ax1.set_xlim([x0 - 8*sigma, x0 + 8*sigma]) 
ax1.set_ylim([y0 - 8*sigma, y0 + 8*sigma]) 
plt.colorbar(im1, ax=ax1) 

ax2 = fg4.add_subplot(332) 
im2 = ax2.pcolormesh(x, x, intensity_0 / np.max(intensity_0), cmap=plt.cm.jet, shading='auto') 
ax2.set_title(f'Measured Intensity of {int(Wavelength * 1e9)} $\pm 5 nm$ in Plane 0', fontsize=12) 
ax2.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax2.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
ax2.set_xlim([x0 - 8*sigma, x0 + 8*sigma]) 
ax2.set_ylim([y0 - 8*sigma, y0 + 8*sigma]) 
plt.colorbar(im1, ax=ax2) 

ax3 = fg4.add_subplot(333) 
im3 = ax3.pcolormesh(x, x, intensity_p / np.max(intensity_p), cmap=plt.cm.jet, shading='auto') 
ax3.set_title(f'Measured Intensity of {int(Wavelength * 1e9)} $\pm 5 nm$ in Plane p', fontsize=12) 
ax3.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax3.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
ax3.set_xlim([x0 - 8*sigma, x0 + 8*sigma]) 
ax3.set_ylim([y0 - 8*sigma, y0 + 8*sigma]) 
plt.colorbar(im1, ax=ax3) 

ax4 = fg4.add_subplot(334) 
im4 = ax4.pcolormesh(x, x, In_updated / np.max(In_updated), cmap=plt.cm.jet, shading='auto') 
ax4.set_title(f'Reconstructed Intensity of {int(Wavelength * 1e9)} $\pm 5 nm$ in Plane n', fontsize=12) 
ax4.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax4.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
ax4.set_xlim([x0 - 8*sigma, x0 + 8*sigma]) 
ax4.set_ylim([y0 - 8*sigma, y0 + 8*sigma]) 
plt.colorbar(im4, ax=ax4) 

ax5 = fg4.add_subplot(335) 
im5 = ax5.pcolormesh(x, x, I0_updated / np.max(I0_updated), cmap=plt.cm.jet, shading='auto') 
ax5.set_title(f'Reconstructed Intensity of {int(Wavelength * 1e9)} $\pm 5 nm$ in Plane 0', fontsize=12) 
ax5.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax5.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
ax5.set_xlim([x0 - 8*sigma, x0 + 8*sigma]) 
ax5.set_ylim([y0 - 8*sigma, y0 + 8*sigma]) 
plt.colorbar(im5, ax=ax5) 

ax6 = fg4.add_subplot(336) 
im6 = ax6.pcolormesh(x, x, Ip_updated / np.max(Ip_updated), cmap=plt.cm.jet, shading='auto') 
ax6.set_title(f'Reconstructed Intensity of {int(Wavelength * 1e9)} $\pm 5 nm$ in Plane p', fontsize=12) 
ax6.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax6.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
ax6.set_xlim([x0 - 8*sigma, x0 + 8*sigma]) 
ax6.set_ylim([y0 - 8*sigma, y0 + 8*sigma]) 
plt.colorbar(im6, ax=ax6) 

ax7 = fg4.add_subplot(337) 
im7 = ax7.pcolormesh(x, x, Phase, cmap=plt.cm.jet, shading='auto') 
ax7.set_title(f'Reconstructed Phase of {int(Wavelength * 1e9)} $\pm 5 nm$', fontsize=12) 
ax7.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax7.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
ax7.set_xlim([x0 - 8*sigma, x0 + 8*sigma]) 
ax7.set_ylim([y0 - 8*sigma, y0 + 8*sigma]) 
plt.colorbar(im7, ax=ax7) 

Error_WA = [Error1, Error2, Error3] 
Label_WA = ['Plane 0', 'Plane n', 'Plane p'] 

ax8 = fg4.add_subplot(326) 
for i in range(len(Error_WA)): 
    ax8.plot(np.linspace(1, iteration, iteration), Error_WA[i], label=Label_WA[i]) 
ax8.set_title('Error History of Gerchberg-Saxton Algorithm', fontsize=12) 
ax8.set_xlabel('Iteration', fontsize=12) 
ax8.set_ylabel('Error', fontsize=12) 
ax8.set_xlim([0, iteration]) 
ax8.legend() 

end = time.time() 
print('"Wavelength Analysis" is Done !') 
print(f'The execution time is {end - start} seconds.') 


###############################################################################
# Spectrum Analysis -- Transverse Focal Shift 
###############################################################################

start = time.time() 
########## ----- Transverse Focal Shift in z=0 ----- ########## 
Wavelength_TFS = np.linspace(left+(10*1e-9), right-(10*1e-9), 51) 
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
ax1.plot(Wavelength_TFS*1e9, Center_x_TFS, marker="o") 
ax1.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax1.set_ylabel('Centroid-x ($m$)', fontsize=12) 
ax1.grid() 

ax2 = fg5.add_subplot(222) 
ax2.plot(Wavelength_TFS*1e9, Center_y_TFS, marker="o") 
ax2.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_ylabel('Centroid-y ($m$)', fontsize=12) 
ax2.grid() 

ax3 = fg5.add_subplot(223) 
ax3.plot(Wavelength_TFS*1e9, Sigma_x_TFS, marker="o") 
ax3.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax3.set_ylabel('Sigma-x ($m$)', fontsize=12) 
ax3.grid() 

ax4 = fg5.add_subplot(224) 
ax4.plot(Wavelength_TFS*1e9, Sigma_y_TFS, marker="o") 
ax4.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax4.set_ylabel('Sigma-y ($m$)', fontsize=12) 
ax4.grid() 

end = time.time() 
print('"Transverse Focal Shift" is Done !') 
print(f'The execution time is {end - start} seconds.') 


###############################################################################
# Spectrum Analysis -- Longitudinal Focal Shift 
###############################################################################

start = time.time() 
########## ----- Longitudinal Focal Shift in z ----- ########## 
Wavelength_LFS = np.array([657, 749, 869]) * 1e-9 
if (len(Wavelength_LFS) > 3): 
    raise ValueError 
for wavelength in Wavelength_LFS: 
    if (wavelength < left + (10*1e-9)) or (wavelength > right - (10*1e-9)): 
        raise ValueError 

Intensity_An_LFS = Decompose_Spectrum(ln, An, Wavelength_LFS) 
Intensity_A0_LFS = Decompose_Spectrum(l0, A0, Wavelength_LFS) 
Intensity_Ap_LFS = Decompose_Spectrum(lp, Ap, Wavelength_LFS) 

Phase_LFS  = [] 
Error_LFS  = [] 
iteration  = 50 
estimation = [0, 0, 0, 0, 0]   # The initial phase estimation coefficients. 
for i in range(len(Wavelength_LFS)): 
    intensity_n = Intensity_An_LFS[i] 
    intensity_0 = Intensity_A0_LFS[i] 
    intensity_p = Intensity_Ap_LFS[i] 
    
    phase, I0_updated, In_updated, Ip_updated, error1, error2, error3 = Gerchberg_Saxton_Algorithm(intensity_n, intensity_0, intensity_p, Wavelength_LFS[i], x, zR, estimation, D, iteration) 
    Phase_LFS.append(phase) 
    Error_LFS.append(error1) 

Distance_LFS  = np.linspace(-0.5, 0.5, 101) * 1e-3 
Intensity_LFS = []   # The intensity of all wavelength. 
Sigma_x_LFS   = []   # The sigma_x of all wavelength. 
Sigma_y_LFS   = []   # The sigma_y of all wavelength. 
Shift_z       = []   # The position of maximum intensity of all wavelength. 
Shift_x_coord = []   # The position of minimum sigma_x of all wavelength. 
Shift_y_coord = []   # The position of minimum sigma_y of all wavelength. 
for i in range(len(Wavelength_LFS)): 
    intensity = Intensity_A0_LFS[i] 
    phase     = Phase_LFS[i] 
    E1        = np.sqrt(intensity) * np.exp(1j*phase) 
    
    intensity_w = []   # The max intensity of all distance for this wavelength. 
    sigma_x_w   = []   # The sigma_x of all distance for this wavelength. 
    sigma_y_w   = []   # The sigma_y of all distance for this wavelength. 
    for distance in Distance_LFS: 
        Eout = Propagator(E1, Wavelength_LFS[i], x, distance) 
        Iout = np.real(Eout * np.conj(Eout)) 
        Iout = Threshold(Iout, 0.1) 
        intensity_w.append(np.max(Iout)) 
        
        sigma_x, sigma_y = Find_Sigma(x, Iout) 
        sigma_x_w.append(sigma_x) 
        sigma_y_w.append(sigma_y) 
    
    Intensity_LFS.append(intensity_w) 
    Sigma_x_LFS.append(sigma_x_w) 
    Sigma_y_LFS.append(sigma_y_w) 
    
    index_max   = np.where(intensity_w == np.max(intensity_w))[0][0] 
    index_min_x = np.where(sigma_x_w == np.min(sigma_x_w))[0][0] 
    index_min_y = np.where(sigma_y_w == np.min(sigma_y_w))[0][0] 
    Shift_z.append(Distance_LFS[index_max]) 
    Shift_x_coord.append(Distance_LFS[index_min_x]) 
    Shift_y_coord.append(Distance_LFS[index_min_y]) 

########## ----- Plot "Longitudinal Focal Shift" in z ----- ########## 
color_list = ['blue','green','red'] 

fg6 = plt.figure(6, figsize=(30, 25)) 
fg6.suptitle('Figure 6 : The Longitudinal Focal Shift in z', fontsize=24) 

ax1 = fg6.add_subplot(421) 
for i in range(len(Wavelength_LFS)): 
    ax1.plot(Distance_LFS, Sigma_x_LFS[i], color=color_list[i], label=f'{Wavelength_LFS[i] * 1e9} nm') 
ax1.set_xlabel('Spatial Distance $z$ ($m$)', fontsize=12) 
ax1.set_ylabel('Sigma-x ($m$)', fontsize=12) 
ax1.legend() 
ax1.grid() 

ax2 = fg6.add_subplot(422) 
ax2.plot(Wavelength_LFS*1e9, Shift_x_coord, marker="o") 
ax2.set_title('Minimum Shift in z of X Coordinate', fontsize=12) 
ax2.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_ylabel('Spatial Distance $z$ ($m$)', fontsize=12) 
ax2.grid() 

ax3 = fg6.add_subplot(423) 
for i in range(len(Wavelength_LFS)): 
    ax3.plot(Distance_LFS, Sigma_y_LFS[i], color=color_list[i], label=f'{Wavelength_LFS[i] * 1e9} nm') 
ax3.set_xlabel('Spatial Distance $z$ ($m$)', fontsize=12) 
ax3.set_ylabel('Sigma-y ($m$)', fontsize=12) 
ax3.legend() 
ax3.grid() 

ax4 = fg6.add_subplot(424) 
ax4.plot(Wavelength_LFS*1e9, Shift_y_coord, marker="o") 
ax4.set_title('Minimum Shift in z of Y Coordinate', fontsize=12) 
ax4.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax4.set_ylabel('Spatial Distance $z$ ($m$)', fontsize=12) 
ax4.grid() 

ax5 = fg6.add_subplot(425) 
for i in range(len(Wavelength_LFS)): 
    ax5.plot(Distance_LFS, Intensity_LFS[i], color=color_list[i], label=f'{Wavelength_LFS[i] * 1e9} nm') 
ax5.set_xlabel('Spatial Distance $z$ ($m$)', fontsize=12) 
ax5.set_ylabel('Intensity', fontsize=12) 
ax5.legend() 
ax5.grid() 

ax6 = fg6.add_subplot(426) 
ax6.plot(Wavelength_LFS*1e9, Shift_z, marker="o") 
ax6.set_title('Minimum Shift in z', fontsize=12) 
ax6.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax6.set_ylabel('Spatial Distance $z$ ($m$)', fontsize=12) 
ax6.grid() 

ax7 = fg6.add_subplot(414) 
for i in range(len(Wavelength_LFS)): 
    ax7.plot(np.linspace(1, iteration, iteration), Error_LFS[i], 
             color=color_list[i], label=f'{Wavelength_LFS[i] * 1e9} nm') 
ax7.set_title('Error History of Gerchberg-Saxton Algorithm', fontsize=12) 
ax7.set_xlabel('Iteration', fontsize=12) 
ax7.set_ylabel('Error', fontsize=12) 
ax7.legend() 
ax7.grid() 

end = time.time() 
print('"Longitudinal Focal Shift" is Done !') 
print(f'The execution time is {end - start} seconds.') 


###############################################################################
# Spectrum Analysis -- Phase Decomposition in Far Field 
###############################################################################

start = time.time() 
########## ----- Phase Decomposition in Far Field ----- ########## 
Wavelength_PD = np.linspace(left+(10*1e-9), right-(10*1e-9), int(((right*1e9-10)-(left*1e9+10))/10 + 2)) 

Intensity_An_PD = Decompose_Spectrum(ln, An, Wavelength_PD) 
Intensity_A0_PD = Decompose_Spectrum(l0, A0, Wavelength_PD) 
Intensity_Ap_PD = Decompose_Spectrum(lp, Ap, Wavelength_PD) 

Phase_FF    = [] 
Error_PD    = [] 
iteration   = 20                # The number of iterations for GS algorithm. 
estimation  = [0, 1, 1, 0, 0]   # The initial phase estimation coefficients. 
Center_x_FF = [] 
Center_y_FF = [] 
Sigma_FF    = [] 
for i in range(len(Wavelength_PD)): 
    intensity_n = Intensity_An_PD[i] 
    intensity_0 = Intensity_A0_PD[i] 
    intensity_p = Intensity_Ap_PD[i] 
    
    phase, I0_updated, In_updated, Ip_updated, error1, error2, error3 = Gerchberg_Saxton_Algorithm(intensity_n, intensity_0, intensity_p, Wavelength_PD[i], x, zR, estimation, D, iteration) 
    Phase_FF.append(phase) 
    Error_PD.append(error1) 
    
    intensity = np.copy(intensity_0) 
    intensity = Threshold(intensity, 0.1) 
    x0, y0    = Find_Centroid(x, Intensity) 
    sigma     = np.max([Find_Sigma(x, intensity)]) 
    Center_x_FF.append(x0) 
    Center_y_FF.append(y0) 
    Sigma_FF.append(sigma) 

N_aberration    = 8   # There are 8 kinds of aberrations. 
Coefficient_FF  = [] 
for i in range(N_aberration): 
    Coefficient_FF.append([]) 

Phase_FF_unwrap = np.copy(Phase_FF) 
for i in range(len(Wavelength_PD)): 
    phase = Phase_FF[i] 
    phase = Phase_Cut(phase, x, Center_x_FF[i], Center_y_FF[i], Sigma_FF[i]) 
    phase = unwrap_phase(phase) 
    coefficient = Phase_Decomposition(phase, x, Center_x_FF[i], Center_y_FF[i], Sigma_FF[i]) 
    for i in range(N_aberration): 
        Coefficient_FF[i].append(coefficient[i]) 

########## ----- Plot "Phase Decomposition in Far Field" ----- ########## 
fg7 = plt.figure(7, figsize=(45, 20)) 
fg7.suptitle('Figure 7 : The Phase Decomposition in Far Field', fontsize=24) 

ax1 = fg7.add_subplot(341) 
ax1.plot(Wavelength_PD*1e9, Coefficient_FF[0], marker="o") 
ax1.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax1.set_ylabel('Tilt Vertical', fontsize=12) 
ax1.grid() 

ax2 = fg7.add_subplot(342) 
ax2.plot(Wavelength_PD*1e9, Coefficient_FF[1], marker="o") 
ax2.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax2.set_ylabel('Tilt Horizontal', fontsize=12) 
ax2.grid() 

ax3 = fg7.add_subplot(343) 
ax3.plot(Wavelength_PD*1e9, Coefficient_FF[2], marker="o") 
ax3.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax3.set_ylabel('Astigmatism Oblique', fontsize=12) 
ax3.grid() 

ax4 = fg7.add_subplot(344) 
ax4.plot(Wavelength_PD*1e9, Coefficient_FF[3], marker="o") 
ax4.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax4.set_ylabel('Astigmatism Horizontal', fontsize=12) 
ax4.grid() 

ax5 = fg7.add_subplot(345) 
ax5.plot(Wavelength_PD*1e9, Coefficient_FF[4], marker="o") 
ax5.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax5.set_ylabel('Defocus', fontsize=12) 
ax5.grid() 

ax6 = fg7.add_subplot(346) 
ax6.plot(Wavelength_PD*1e9, Coefficient_FF[5], marker="o") 
ax6.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax6.set_ylabel('Coma Vertical', fontsize=12) 
ax6.grid() 

ax7 = fg7.add_subplot(347) 
ax7.plot(Wavelength_PD*1e9, Coefficient_FF[6], marker="o") 
ax7.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax7.set_ylabel('Coma Horizontal', fontsize=12) 
ax7.grid() 

ax8 = fg7.add_subplot(348) 
ax8.plot(Wavelength_PD*1e9, Coefficient_FF[7], marker="o") 
ax8.set_xlabel('Wavelength ($nm$)', fontsize=12) 
ax8.set_ylabel('Spherical', fontsize=12) 
ax8.grid() 

ax9 = fg7.add_subplot(313) 
for i in range(len(Wavelength_PD)): 
    ax9.plot(np.linspace(1, iteration, iteration), Error_PD[i], label=f'{Wavelength_PD[i] * 1e9} nm') 
ax9.set_title('Error History of Gerchberg-Saxton Algorithm')
ax9.set_xlabel('Iteration', fontsize=12) 
ax9.set_ylabel('Error', fontsize=12) 
ax9.legend() 
ax9.grid() 

end = time.time() 
print('"Phase Decomposition in Far Field" is Done !') 
print(f'The execution time is {end - start} seconds.') 


###############################################################################
# Spectrum Analysis -- Near Field Analysis 
###############################################################################

start = time.time() 
########## ----- Near Field Analysis ----- ########## 
E0_all = 0    # The total wave in the near field. 
E0_idv = []   # The wave in the near field for different wavelength. 
for i in range(len(Wavelength_PD)): 
    E1 = np.sqrt(Intensity_A0_PD[i]) * np.exp(1j*Phase_FF[i]) 
    E0 = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(E1))) 
    E0_idv.append(E0) 
    E0_all += E0 

Intensity_NF = np.real(E0_all * np.conj(E0_all)) 
intensity    = np.copy(Intensity_NF) 
intensity    = Threshold(intensity, 0.1) 
x0, y0       = Find_Centroid(x, Intensity) 
sigma        = np.max([Find_Sigma(x, intensity)]) 

N   = np.size(x) 
S1  = 1 / (x[1] - x[0])                       # The size of object plane in $m$. 
ds1 = S1/N                                    # The size of one pixel. 
x1  = np.linspace(-(N/2)*ds1, (N/2)*ds1, N)   # The axis in object plane. 

Phase_NF = np.angle(E0_all) 
Phase_NF = Phase_Cut(Phase, x1, x0, y0, sigma) 
Phase_NF = unwrap_phase(Phase_NF) 

########## ----- Plot "Near Field Analysis" ----- ########## 
fg8 = plt.figure(8, figsize=(25, 10)) 
fg8.suptitle('Figure 8 : The Near Field Analysis', fontsize=24) 

ax1 = fg8.add_subplot(121) 
im1 = ax1.pcolormesh(x1, x1, Intensity_NF, cmap=plt.cm.jet, shading='auto') 
ax1.set_title('Reconstructed Intensity in Near Field', fontsize=12) 
ax1.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax1.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
plt.colorbar(im1, ax=ax1) 

ax2 = fg8.add_subplot(122) 
im2 = ax2.pcolormesh(x1, x1, Phase_NF, cmap=plt.cm.jet, shading='auto') 
ax2.set_title('Reconstructed Phase in Near Field', fontsize=12) 
ax2.set_xlabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax2.set_ylabel('Spatial Distance $y$ ($m$)', fontsize=12) 
plt.colorbar(im2, ax=ax2) 

end = time.time() 
print('"Near Field Analysis" is Done !') 
print(f'The execution time is {end - start} seconds.') 


X1, Y1  = np.meshgrid(x1, x1) 
radius1 = np.sqrt((X1-x0)**2 + (Y1-y0)**2) / (2*np.sqrt(2)*sigma) 
theta1  = np.arctan2((Y1-y0), (X1-x0)) 