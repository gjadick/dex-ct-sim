#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:21:03 2023

@author: giavanna
"""


import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('xtomosim')  
import xtomosim.xcompy as xc
from matdecomp import matcomp1, matcomp2


# some plotting params
figdir = 'output/mvkv-pcd/figs/'
fmt = 'png'
savefig = True
os.makedirs(figdir, exist_ok=True)

# input data?  for noise = 10 photons
std_e = 10
phantom = 'pelvis'
outdir = 'output/mvkv-pcd/'
matdecompdir = 'matdecomp_detunedMV_80kV_9000uGy_1000uGy/'
modepaths = {'EID ($\sigma_e$ = 10)': f'{outdir}/{phantom}_eid_0010std/',
             'EID': f'{outdir}/{phantom}_eid_noiseless/',
             'PCD': f'{outdir}/{phantom}_pcd_noiseless/'
             }
modes = list(modepaths.keys())

# load some of the recons
N = 512
imgs_mat1 = []
imgs_mat2 = []
for mode in modes:
    imdir = modepaths[mode] + matdecompdir
    recon_mat1 = np.fromfile(imdir + 'mat1_recon_float32.bin', dtype=np.float32).reshape([N,N])
    recon_mat2 = np.fromfile(imdir + 'mat2_recon_float32.bin', dtype=np.float32).reshape([N,N])
    imgs_mat1.append(recon_mat1)
    imgs_mat2.append(recon_mat2)
    

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


matname1 = 'tissue'
matname2 = 'bone'
matnames = [matname1, matname2]
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


#%% FIG : Detector models

# The chemical weights for each matcomp were found from NIST XCOM.
# https://physics.nist.gov/cgi-bin/Xcom/xcom2?Method=Mix&Output2=Hand

# MV detector
matcomp_MV = 'Mo'
p_MV = 10.22
t_MV = 0.60
det_MV = ['MV EID', matcomp_MV, p_MV, t_MV]

# Silicon PCD
matcomp_Si = 'Si'
p_Si = 2.33
t_Si = 3.0
det_Si = [f'{10*t_Si:.0f}-mm Si', matcomp_Si, p_Si, t_Si]

# CdTe PCD
matcomp_CdTe = 'Cd(46.8358)Te(53.1642)'
p_CdTe = 5.85
t_CdTe = 0.2
det_CdTe = [f'{10*t_CdTe:.0f}-mm CdTe', matcomp_CdTe, p_CdTe, t_CdTe]

# CZT PCD
matcomp_CZT = 'Cd(36.8077)Zn(21.4112)Te(41.7811)'
p_CZT = 5.85  # ~5.9
t_CZT = 0.2
det_CZT = [f'{10*t_CZT:.0f}-mm CZT', matcomp_CZT, p_CZT, t_CZT]

dets = [det_MV, det_Si, det_CdTe, det_CZT]

fig, ax = plt.subplots(1,2,figsize=[5,2.5])
for axi, energy_units, E in zip(ax, ['keV','MeV'], [np.linspace(1, 140, 2000), np.linspace(200, 6000, 2000)]):
    axi.set_prop_cycle(color =     ['r', 'b', 'k', 'orange'], 
                       linestyle = ['-', '-', '-', '--']) 
    for name, matcomp, density, thickness in dets:

        mu = xc.mixatten(matcomp, E) * density
        eta = 1 - np.exp(-mu * thickness)
        if energy_units=='keV':
            axi.plot(E, eta, label=name)
        elif energy_units=='MeV':
            axi.plot(E*1e-3, eta)#, label=name)
    axi.set_title(f'{energy_units} scale')
    axi.set_xlabel(f'x-ray energy [{energy_units}]')
ax[0].set_ylabel('detective efficiency')
fig.tight_layout()
fig.legend(framealpha=1, fontsize=8, bbox_to_anchor=(1.2,0.7), title='Detector')
label_panels(ax, dy=0.08)

if savefig:
    plt.savefig(figdir+f'detectors.{fmt}', bbox_inches='tight')
plt.show()


#%% FIG : basis material images
Nx, Ny = 350, 350
for j, data in enumerate([[imgs_mat1, 0.95, 0.3],
                          [imgs_mat2, 1.5, 1]
                         ]):
    basismat_imgs, WL, WW = data
    fig, ax = plt.subplots(1, 3, figsize=[7, 2.5])
    for i in range(3):
        kw = {'vmax':WL + WW/2, 'vmin':WL - WW/2}
        bmi = basismat_imgs[i]
        bmi = bmi[int((N-Ny)/2):int((N+Ny)/2), int((N-Nx)/2):int((N+Nx)/2)]
        m = ax[i].imshow(bmi, aspect='equal', **kw)
        ax[i].axis('off')
        ax[i].text(10, 10, f'WL = {WL:.1f} g/cm$^3$ \nWW= {WW:.1f} g/cm$^3$',
                   fontsize='small', color='w', ha='left', va='top')
        ax[i].set_title(modes[i])
    fig.tight_layout()
    label_panels(ax, dy=0.06)
    if savefig:
        plt.savefig(figdir+f'imgs_{matnames[j]}.{fmt}', bbox_inches='tight')
    plt.show()


#%% FIG : virtual mono images

WL, WW = 30, 200
Nx, Ny = 350, 350
for E0 in [50, 100, 200]:
    fig, ax = plt.subplots(1, 3, figsize=[7, 2.5])
    for i in range(3):
        kw = {'vmax':WL + WW/2, 'vmin':WL - WW/2}
        vmi_E0 = make_vmi(E0, imgs_mat1[i], imgs_mat2[i])
        vmi_E0 = vmi_E0[int((N-Ny)/2):int((N+Ny)/2), int((N-Nx)/2):int((N+Nx)/2)]
        m = ax[i].imshow(vmi_E0, aspect='equal', **kw)
        ax[i].axis('off')
        ax[i].text(10, 10, f'E$_0$ = {E0:.0f} keV \nWL = {WL:.0f} \nWW= {WW:.0f}', 
                   fontsize='small', color='w', ha='left', va='top')
        ax[i].set_title(modes[i])
    fig.tight_layout()
    label_panels(ax, dy=0.06)
    if savefig:
        plt.savefig(figdir+f'vmi_{int(E0):03}keV.{fmt}', bbox_inches='tight')
    plt.show()


#%% FIG : show ROIs for CNR measurements

dx, dy, x0, y0 = 20, 20, 260, 256  
dx2, dy2, x02, y02 = 20, 20, 290, 300
Nx, Ny = 350, 350  # size of cropped image
roi_sig = np.array([int(x0-(N-Nx)/2 - dx/2),  int(y0-(N-Ny)/2 - dy/2),  dx,  dy])  # transform ROIs to crop
roi_bg = np.array([ int(x02-(N-Nx)/2 - dx2/2), int(y02-(N-Ny)/2 - dy2/2), dx2, dy2])

for E0 in [70]:
    fig, ax = plt.subplots(1, 3, figsize=[7, 2.5])
    for i in range(3):
        kw = {'vmax':WL + WW/2, 'vmin':WL - WW/2}
        
        vmi_E0 = make_vmi(E0, imgs_mat1[i], imgs_mat2[i])
        vmi_E0 = vmi_E0[int((N-Ny)/2):int((N+Ny)/2), int((N-Nx)/2):int((N+Nx)/2)]

        # show ROIs
        u1, v1 = measure_roi(vmi_E0, roi_sig, ax=ax[i]) # signal
        u2, v2 = measure_roi(vmi_E0, roi_bg, ax=ax[i])  # background
        cnr = np.abs(u1 - u2) / np.sqrt(v1 + v2)
        
        # transform ROIs to crop
        m = ax[i].imshow(vmi_E0, aspect='equal', **kw)
        ax[i].axis('off')
        ax[i].text(10, 10, f'E$_0$ = {E0:.0f} keV \nWL = {WL:.0f} \nWW= {WW:.0f}', 
                   fontsize='small', color='w', ha='left', va='top')
        ax[i].text(Nx-10, 10, f'CNR = {cnr:.2f}', 
                   fontsize='small', color='w', ha='right', va='top')
        ax[i].set_title(modes[i])
    fig.tight_layout()
    label_panels(ax, dy=0.06)
    if savefig:
        plt.savefig(figdir+f'xcat_roi_{int(E0):03}keV.{fmt}', bbox_inches='tight')
    plt.show()



#%% FIG : CNR vs. E0 (use ROIs above)
    
    modekw = {'PCD': {'label':'PCD', 'color':'b', 'linestyle':'-' },
          'EID': {'label':'EID (noiseless)', 'color':'k', 'linestyle':'-' },
          'EID ($\\sigma_e$ = 10)': {'label':'EID ($\sigma_e = $'+f' {int(std_e)} photons)', 'color':'r', 'linestyle':'--' }
         }
        
    # use ROIs above, but without cropping
    roi_sig = np.array([int(x0-dx/2),  int(y0-dy/2),  dx,  dy])  # transform ROIs to crop
    roi_bg = np.array([ int(x02-dx2/2), int(y02-dy2/2), dx2, dy2])
    
    monoenergies = np.linspace(30, 150, 50)
    fig, ax = plt.subplots(1, 1, figsize=[5, 3])
    for i in range(3):
        cnr_vec = []
        for E0 in monoenergies:
            
            vmi_E0 = make_vmi(E0, imgs_mat1[i], imgs_mat2[i])
            u1, v1 = measure_roi(vmi_E0, roi_sig) # signal
            u2, v2 = measure_roi(vmi_E0, roi_bg)  # background
            cnr_E0 = np.abs(u1 - u2) / np.sqrt(v1 + v2)
            cnr_vec.append(cnr_E0)        
    
        ax.plot(monoenergies, cnr_vec, **modekw[modes[i]])
    ax.legend()
    ax.set_ylabel('CNR')
    ax.set_xlabel('VMI energy [keV]')
    fig.tight_layout()
    if savefig:
        plt.savefig(figdir+f'cnr.{fmt}', bbox_inches='tight')
    plt.show()





