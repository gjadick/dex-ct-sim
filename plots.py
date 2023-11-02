#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 07:50:57 2023

@author: gjadick
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import os

import sys
sys.path.append('xtomosim')  
import xtomosim.xcompy as xc
from xtomosim.system import FanBeamGeometry, VoxelPhantom
from matdecomp import mat1, mat2, matcomp1, matcomp2


# some plotting params
figdir = 'output/figs/mvkv_r1/'
savefig = True
os.makedirs(figdir, exist_ok=True)

plt.rcParams.update({
    # figure
    "figure.dpi": 600,
    # text
    "font.size":10,
    "font.family": "serif",
    "font.serif": ['Computer Modern Roman'],
    "text.usetex": True,
    # axes
    "axes.titlesize":10,
    "axes.labelsize":8,
    "axes.linewidth": 1,
    # ticks
    "xtick.top": True, 
    "ytick.right": True, 
    "xtick.direction": "in", 
    "ytick.direction": "in",
    "xtick.labelsize":8,
    "ytick.labelsize":8,
    # grid
    "axes.grid" : False, 
     "grid.color": "lightgray",
     "grid.linestyle": ":",
     # legend
     "legend.fontsize":8,
     # lines
     #"lines.markersize":5,
     "lines.linewidth":1,
     })


def bf(string):  
    '''make text boldface'''
    return "\\textbf{"+string+"}"


def label_panels(ax, c='k', loc='outside', dx=-0.06, dy=0.09, fontsize=None,
                 label_type='lowercase', label_format='({})'):
    '''
    Function to label panels of multiple subplots in a single figure.

    Parameters
    ----------
    ax : matplotlib AxesSubplot
    c : (str) color of text. The default is 'k'.
    loc : (str), location of label, 'inside' or 'outside'. 
    dx : (float) x location relative to upper left corner. The default is 0.07.
    dy : (float) y location relative to upper left corner. The default is 0.07.
    fontsize : (number), font size of label. The default is None.
    label_type : (str), style of labels. The default is 'lowercase'.
    label_format : (str) format string for label. The default is '({})'.

    '''
    if 'upper' in label_type:
        labels = list(map(chr, range(65,91)))
    elif 'lower' in label_type:
        labels = list(map(chr, range(97, 123)))
    else: # default to numbers
        labels = np.arange(1,27).astype(str)
    labels = [ label_format.format(x) for x in labels ]

    # get location of text
    if loc == 'outside':
        xp, yp = -dx, 1+dy
    else:
        xp, yp = dx, 1-dy
        
    for i, axi in enumerate(ax.ravel()):
        xmin, xmax = axi.get_xlim()
        ymin, ymax = axi.get_ylim()
        xloc = xmin + (xmax-xmin)*xp
        yloc = ymin + (ymax-ymin)*yp
        label = labels[i]
        axi.text(xloc, yloc, bf(label), color=c, fontsize=fontsize,
          va='center', ha='center')
        
    return None


        
#%% stuff for plotting 

# acquisition geometry
ct = FanBeamGeometry(N_channels=800, N_proj=1200, gamma_fan=0.8230337,  
                  SID=60.0, SDD=100.0, h_iso=1.0, 
                  eid=True, detector_file='./input/detector/eta_eid_mv.bin')  

# some recon params
N_matrix = 512
r0 = N_matrix//2  # for centering
FOV = 50.0
ramp = 0.8

# phantoms
proot = 'input/phantom/mvkv/xcat_'
def pname(txt):
    return proot + txt + '_uint8_512_512_1_1mm.bin'
phantom_dict = {
    'pelvis' : VoxelPhantom('pelvis', pname('pelvis'), proot+'materials.csv', 512, 512, 1, z_index=0), 
    'pelvis_steel' : VoxelPhantom('pelvis_steel', pname('pelvis_metal'), proot+'materials.csv', 512, 512, 1, z_index=0),
    'pelvis_titanium' : VoxelPhantom('pelvis_titanium', pname('pelvis_metal'), proot+'materials_titanium.csv', 512, 512, 1, z_index=0) 
    }
phantoms = phantom_dict.keys()

# pairs to iterate over for plotting
# spec_id1,  spec_id2, dose1, dose2
spec_pairs = [ ['140kV',    '80kV',   5,     5],
               ['detunedMV','80kV',   9,     1] ]


def make_vmi(E0, M1, M2, 
             HU=True, matcomp1=matcomp1, matcomp2=matcomp2):
    u_p_1 = xc.mixatten(matcomp1, np.array([E0]).astype(np.float64))
    u_p_2 = xc.mixatten(matcomp2, np.array([E0]).astype(np.float64))
    u_w =  1.0 * xc.mixatten('H(11.2)O(88.8)',  np.array([E0]).astype(np.float64))
    vmi = u_p_1*M1 + u_p_2*M2
    if HU:
        vmi = 1000*(vmi-u_w)/u_w
    return vmi.astype(np.float32)

def measure_roi(M, roi_info, give_roi=False, ax=None):
    x0, y0, dx, dy = roi_info
    mask = np.zeros(M.shape, dtype=bool)
    mask[y0:y0+dy, x0:x0+dx] = 1
    ROI = M[mask]
    u, v = np.mean(ROI), np.var(ROI)
    if ax is not None:
        xvals = [x0+dx, x0, x0, x0+dx, x0+dx]
        yvals = [y0, y0, y0+dy, y0+dy, y0]
        ax.plot(xvals, yvals, 'r-', lw=0.5)
    if give_roi:
        return ROI
    return u, v

def make_cbar(axi, m, label, 
              cbar_fsz=8, cbar_pad=0.01, cbar_lab_pad=-5):
    cbar = fig.colorbar(m, ax=axi, pad=cbar_pad)
    cbar.ax.tick_params(labelsize=cbar_fsz) 
    cbar.set_label(label, labelpad=cbar_lab_pad)

def crop_img(M, crop):
    r0 = M.shape[0]//2
    M_cropped = M[r0-crop//2:r0+crop//2, r0-crop//2:r0+crop//2]
    return M_cropped

def get_img_ct(phantom_id, spec_id, dose, crop=None, units='HU',
               N_matrix=N_matrix, FOV=FOV, ramp=ramp):
    assert units=='HU' or units=='raw'
    img_dir = f'output/mvkv_{phantom_id}/{spec_id}_{int(dose*1000):04}uGy/'     
    fname = f'recon_{units}_float32.bin' 
    M = np.fromfile(img_dir+fname, dtype=np.float32).reshape([N_matrix, N_matrix])
    if crop is not None:
        M = crop_img(M, crop)
    return M

def get_img_basismats(phantom_id, spec_id1, spec_id2, dose1, dose2, crop=None,
                      N_matrix=N_matrix, FOV=FOV, ramp=ramp):
    img_dir = f'output/mvkv_{phantom_id}/matdecomp_{spec_id1}_{spec_id2}_{int(dose1*1000):04}uGy_{int(dose2*1000):04}uGy/'     
    M1 = np.fromfile(img_dir+'mat1_recon_float32.bin', dtype=np.float32).reshape([N_matrix, N_matrix])
    M2 = np.fromfile(img_dir+'mat2_recon_float32.bin', dtype=np.float32).reshape([N_matrix, N_matrix])
    if crop is not None:
        M1 = crop_img(M1, crop)
        M2 = crop_img(M2, crop)
    return M1, M2

def register_xcat(M0, N0=512, Nf=524, dx=6, dy=6):
    '''
    Register original XCAT to the reconstructed images.
    Scale parameters (N0, Nf) chosen by known pixel sz difference,
    shift parameters (dx, dy) chosen by visual inspection.
    '''
    X0 = np.linspace(0, N0, N0)  # old pts
    interp = RegularGridInterpolator((X0, X0), M0)
    
    X = np.linspace(0, N0, Nf)   # new pts
    x, y = np.meshgrid(X, X, indexing='ij')
    
    M = interp((x,y))         # scale
    M = M[dy:dy+N0,dx:dx+N0]  # shift
    
    return M

def get_xcat_mask(M, threshold=-900):
    ''' get mask of values above threshold
    default threshold set to mask non-air pixels '''
    mask = np.zeros(M.shape, dtype=bool)
    mask[M>threshold] = True
    return mask


#%% Location of ROIs for different measurements

# coords of ROIs for SNR measurements, dx, dy, x0, y0, x02, y02
# x0, y0 gives signal and x02, y02 gives background
# these were manually selected
dx, dy, x0, y0 = 25, 25, 248, 246  
dx2, dy2, x02, y02 = 20, 20, 282, 286
 
WL, WW = 100, 500
hu_kwargs = {'cmap':'gray', 'vmin':WL-WW/2, 'vmax':WL+WW/2}

fig, ax = plt.subplots(1,2, dpi=300, figsize=[6.5,2] )

for i, phantom in enumerate(['pelvis', 'pelvis_steel']):
    
    # plot xcat
    xcat = phantom_dict[phantom]
    M = register_xcat(xcat.M_mono(80).get())
    ax[i].imshow(M, **hu_kwargs)
    ax[i].axis('off')
    ax[i].set_title(phantom.replace('_SS', ' with metal'))

    # crop the image and add text with W/L params
    Y0 = 140  
    ax[i].set_ylim(400, Y0)
    ax[i].text(5,Y0+5,f'W/L = {WW}/{WL}',color='w', ha='left', va='top')
    
    # show ROIs
    _, _ = measure_roi(M, [x0,  y0, dx, dy], ax=ax[i])   # signal
    _, _ = measure_roi(M, [x02, y02, dx2, dy2], ax=ax[i])  # background

ax[1].arrow(110, 212, 38, 25, facecolor='yellow', edgecolor='k',
            width=15, head_width=35, head_length=20)
label_panels(ax, loc='outside')
fig.tight_layout()
if savefig:
    plt.savefig(figdir+'phantom_rois.pdf')
plt.show()
    



#%% RMSE measurements for pelvis w/w/o metal

fig, ax = plt.subplots(1,3, dpi=300, figsize=[8, 2.5])
phantnames = f'_{int(100*ramp)}ramp'
legend_elements = []

for i, [phantom, Evals] in enumerate([
              ['pelvis',       np.arange(40, 140, 10)], 
              ['pelvis_titanium',  np.arange(100, 320, 24)], 
              ['pelvis_steel',  np.arange(100, 320, 24)],  
             ]):  # 20
    ax[i].set_title(phantom.replace('_', ' with '))
    
    xcat = phantom_dict[phantom]
    mask = get_xcat_mask(register_xcat(xcat.M_mono(120).get()))  # mask of non-air values

    for spec1, spec2, D1, D2, ls, sid in [
            ['detunedMV', '80kV', 9, 1, 'ro', 'MV-80kV'],
            ['140kV',     '80kV', 5, 5, 'bs', '140-80kV']]:
        M_m1, M_m2 = get_img_basismats(phantom, spec1, spec2, D1, D2)#, crop=N_crop)
        
        rmses = []
        Evals_all = np.arange(Evals[0], Evals[-1]+1, 1)  # get smooth curve
        for E0 in Evals_all:
            vmi = make_vmi(E0, M_m1, M_m2)[mask]
            vmi_gt = register_xcat(xcat.M_mono(E0).get())[mask]
            rmse = np.sqrt( np.mean( (vmi-vmi_gt)**2 ) )
            rmses.append(rmse)
            
        # plot and print the minimum, for calculating percent difference
        print(phantom, sid, f'min RMSE = {np.min(rmses):.3f} @ {Evals_all[np.argmin(rmses)]} keV')
        ax[i].plot(Evals_all, rmses, ls[0]+'-')
        # add markers to identify curves in black and white
        ax[i].plot(Evals, rmses[::Evals[1]-Evals[0]], ls, markerfacecolor="None")
        
        if i==0:   # add label to legend once
            legend_elements.append( plt.Line2D([0], [0], color=ls[0], marker=ls[1], markerfacecolor='None', label=sid) )

ax[0].set_ylabel('RMSE [HU]')
for axi in ax:
    axi.set_xlabel('VMI energy [keV]')
    lx0, lx1 = axi.get_xlim()
    ly0, ly1 = axi.get_ylim()
    axi.set_aspect(0.9*(lx1-lx0)/(ly1-ly0))
    
fig.tight_layout(pad=1.1)
fig.legend(handles=legend_elements, loc='center right')
plt.subplots_adjust(right = 0.86)
label_panels(ax, dy=0.06)

if savefig:
    plt.savefig(figdir+f'rmse{phantnames}.pdf')
plt.show()
    


#%% CNR measurements for pelvis w/w/o metal

fig, ax = plt.subplots(1,3, dpi=300, figsize=[8.5, 2.8])
ax[0].set_ylabel('CNR')

phantnames = f'_{int(100*ramp)}ramp'
legend_elements = []

for i, [phantom, Evals] in enumerate([
              ['pelvis',       np.arange(40, 140, 10)], 
              ['pelvis_titanium',  np.arange(100, 320, 24)], 
              ['pelvis_steel',  np.arange(100, 320, 24)],  
            ]):
    ax[i].set_title(phantom.replace('_', ' with '))
    
    xcat = phantom_dict[phantom]

    # single specs
    tlw, col = 1.5, 'k'
    for spec, D, ls in  [['80kV', 10, '--'], 
                        ['120kV', 10, ':' ], 
                        ['140kV', 10, '-' ]]:

        M = get_img_ct(phantom, spec, D)
        u1, v1 = measure_roi(M, [x0, y0, dx, dy])#, ax=ax[i])   # signal
        u2, v2 = measure_roi(M, [x02, y02, dx, dy])#, ax=ax[i]) # background
        snr = (u1-u2)/np.sqrt(v1 + v2)
        print(phantom, spec, f'SNR = {snr:.3f}')
        ax[i].axhline(snr, lw=tlw, color=col, ls=ls, label=spec)
        if i==0:
            legend_elements.append(plt.Line2D([0], [0], color=col, ls=ls,  label=spec))

    # dual specs
    for spec1, spec2, D1, D2, ls, sid in [
            ['140kV',     '80kV', 5, 5,  'bs', '140-80kV'],
            ['detunedMV', '80kV', 9, 1,  'ro', 'MV-80kV']]:
        
        M_m1, M_m2 = get_img_basismats(phantom, spec1, spec2, D1, D2)
        
        Evals_all = np.arange(Evals[0], Evals[-1]+1, 1)  # get smooth curve
        snrs = []
        for E0 in Evals_all:
            vmi = make_vmi(E0, M_m1, M_m2)
            u1, v1 = measure_roi(vmi, [x0, y0, dx, dy])#, ax=ax[i])   # signal
            u2, v2 = measure_roi(vmi, [x02, y02, dx, dy])#, ax=ax[i]) # background
            snr = (u1-u2)/np.sqrt(v1 + v2)
            snrs.append(snr)
             
        # plot and print the maximum
        print(phantom, sid, f'max SNR = {np.max(snrs):.3f} @ {Evals_all[np.argmax(snrs)]} keV')
        ax[i].plot(Evals_all, snrs, ls[0]+'-')
        # add markers to identify curves in black and white
        ax[i].plot(Evals, snrs[::Evals[1]-Evals[0]], ls, markerfacecolor="None")
        
        if i==0:
            legend_elements.append( plt.Line2D([0], [0], color=ls[0], marker=ls[1], markerfacecolor='None', label=sid) )

for axi in ax:
    axi.set_xlabel('VMI energy [keV]')
    tx0, tx1 = axi.get_xlim()
    ty0, ty1 = axi.get_ylim()
    axi.set_aspect(0.9*(tx1-tx0)/(ty1-ty0))

fig.tight_layout(pad=1.1)
fig.legend(handles=legend_elements, loc='center right')
plt.subplots_adjust(right=0.86)  
label_panels(ax, dy=0.06)

if savefig:
    plt.savefig(figdir+f'snr{phantnames}.pdf')
plt.show()



#%% All images per DE-CT acquisition (SE-CTs, BMIs, VMIs)

E0s = [80, 300]

for phantom, WL, WW, N_crop in [
            ['pelvis', 50, 500, 380],
            ['pelvis_titanium', 50, 500 , 380 ],
            ['pelvis_steel', 50, 500 , 380 ],
            ]:
    hu_kwargs = {'cmap':'gray', 'vmin':WL-WW/2, 'vmax':WL+WW/2}
    mat1_kwargs = {'cmap':'gray', 'vmin':0, 'vmax':1.2}  # turbo
    mat2_kwargs = {'cmap':'gray', 'vmin':0, 'vmax':2.2}

    for spec1, spec2, D1, D2 in spec_pairs:
        
        fig, ax = plt.subplots(3,2, dpi=300, figsize=[6.3,8])

        M_s1 = get_img_ct(phantom, spec1, D1, crop=N_crop)
        M_s2 = get_img_ct(phantom, spec2, D2, crop=N_crop)
        M_m1, M_m2 = get_img_basismats(phantom, spec1, spec2, D1, D2, crop=N_crop)
        vmi1 = make_vmi(E0s[0], M_m1, M_m2)
        vmi2 = make_vmi(E0s[1], M_m1, M_m2)
        
        # COLUMN 0 - raw images
        ax[0,0].set_title(f'{spec1} ({D1}mGy)')
        m = ax[0,0].imshow(M_s1, **hu_kwargs)
        make_cbar(ax[0,0], m, 'HU')

        ax[0,1].set_title(f'{spec2} ({D2}mGy)')
        m = ax[0,1].imshow(M_s2, **hu_kwargs)
        make_cbar(ax[0,1], m, 'HU')

        # COLUMN 2 - basis mats
        ax[1,0].set_title(f'BMI - {mat1}')
        m = ax[1,0].imshow(M_m1, **mat1_kwargs)
        make_cbar(ax[1,0], m, '\n$\\rho$ [g/cm$^3$]')

        ax[1,1].set_title(f'BMI - {mat2}')
        m = ax[1,1].imshow(M_m2, **mat2_kwargs)
        make_cbar(ax[1,1], m, '\n$\\rho$ [g/cm$^3$]')
        
        # COLUMN 3 - VMIs
        ax[2,0].set_title(f'VMI - {E0s[0]} keV')
        m = ax[2,0].imshow(vmi1, **hu_kwargs)
        make_cbar(ax[2,0], m, f'HU [{E0s[0]} keV]')

        ax[2,1].set_title(f'VMI - {E0s[1]} keV')
        m = ax[2,1].imshow(vmi2, **hu_kwargs)
        make_cbar(ax[2,1], m, f'HU [{E0s[1]} keV]')

        for axi in ax.ravel():
            axi.axis('off')
            
        fig.tight_layout(pad=0.1)
        label_panels(ax, dy=0.05)
        
        if savefig:
            fname = f'{phantom}_{spec1}_{spec2}_{int(D1)}mGy_{int(D2)}mGy_ramp{int(100*ramp)}_{int(E0s[0])}keV_{int(E0s[1])}keV.pdf'
            plt.savefig(figdir+fname)
        plt.show()



#%% Revision 1 : metal attenuation

data_SS = [['Steel 316L', 8.0, 'C(0.5)N(0.1)P(0.0025)S(0.01)Fe(64.335)Cr(17.0)Ni(13.0)Mo(2.25)Mn(2.0)Si(0.75)Cu(0.5)']]

# 2) Ti and alloys
data_Ti = [['Pure Ti', 4.5, 'Ti(100.0)'],
           ['Ti-6Al-4V', 4.43, 'Al(6)Ti(90)V(4)'],
           ['Ti-5Al-2.5Fe', 4.49, 'Ti(92.5)Al(5)Fe(2.5)'],
           ['Ti-6Al-7Nb ', 4.49, 'Ti(87)Al(6)Nb(7)']]

# 3) Co alloys
data_CoCr = [['Co-28Cr-6Mo', 8.5, 'Co(66)Cr(28)Mo(6)'],  # F75 (-98)   -- same comp as F799-99
             ['Co-35Ni-20Cr-10Mo', 8.5,'Co(35)Ni(35)Cr(20)Mo(10)'],  # F562 (-95)
             ['Co-20Cr-15W-10Ni', 8.5, 'Co(55)Cr(20)W(15)Ni(10)']]  # F90 (-97)

lines = ['-', '--', ':', '-.', '-']
datas = [data_CoCr, data_Ti, data_SS]
subcols = [
           ['firebrick','tomato','darkorange'], 
           ['darkblue', 'blue', 'cornflowerblue','steelblue','lightsteelblue'], 
           ['k', 'gray', 'silver'],]

fig, ax = plt.subplots(1,2,figsize=[5.3,2.6])
for axi, energy_units, E in zip(ax, ['keV','MeV'], [np.linspace(1, 140, 2000), np.linspace(150, 6500, 2000)]):
    for j, data in enumerate(datas):
        for i, x in enumerate(data):
            ls = lines[i]
            col = subcols[j][i]
            name, density, matcomp = x
            mu = xc.mixatten(matcomp, E)*density
            if energy_units=='keV':
                axi.plot(E, mu, ls=ls, lw=1., color=col, label=name)
            elif energy_units=='MeV':
                axi.plot(E*1e-3, mu, ls=ls, lw=1., color=col)#, label=name)
    axi.set_title(f'{energy_units}-scale')
    axi.set_yscale('log')
    axi.set_xlabel(f'energy [{energy_units}]')
    #axi.legend()
ax[0].set_ylabel('linear attenuation coefficient [cm$^{-1}$]')
fig.tight_layout(pad=0.5)
fig.legend(framealpha=1, fontsize=8, bbox_to_anchor=(1.04,0.9))#loc='center right')#title='energy, b')

# manual panel labeling because of y-axis logscale
#label_panels(ax, dy=0.25)
ax[0].text(3.224, 3e5, bf('(a)'), va='center', ha='center')
ax[1].text(.2516, 5.1, bf('(b)'), va='center', ha='center')

if savefig:
    plt.savefig(figdir+'metal_LACs.pdf', bbox_inches='tight')
plt.show()



