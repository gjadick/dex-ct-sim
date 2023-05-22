#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 07:50:57 2023

@author: gjadick
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

from main import get_out_dir_se, get_out_dir_de
from system import ScannerGeometry, Phantom, Spectrum
from matdecomp import mat1, mat2, matcomp1, matcomp2
import xcompy as xc


#%%

# some plotting params
figdir = 'output/figs/'

plt.rcParams.update({
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
    "axes.grid" : True, 
     "grid.color": "lightgray",
     "grid.linestyle": ":",
     # legend
     "legend.fontsize":8,
     # lines
     #"lines.markersize":5,
     "lines.linewidth":1,
     })


        
#%% stuff for plotting 

# acquisition geometry
ct = ScannerGeometry(N_channels=800, N_proj=1200, gamma_fan=0.8230337,  
                  SID=60.0, SDD=100.0, pxshape='rect')  

# some recon params
N_matrix = 512
r0 = N_matrix//2  # for centering

# spectra 
filepath_spectrum = 'input/spectrum/'
filename_dict = {
    '6MV':       "Accuray_treatment6MV.csv", 
    'detunedMV': "Accuray_detuned.csv", 
    '80kV':      "spec80.mat", 
    '120kV':     "spec120.mat", 
    '140kV':     "spec140.mat"     }
spec_dict = {}
for spec_id in filename_dict:
    spec = Spectrum(filepath_spectrum+filename_dict[spec_id], spec_id)
    spec_dict[spec_id] = spec
    
    
# pairs to iterate over for plotting
# spec_id1,  spec_id2, dose1, dose2
spec_pairs = [ ['140kV',    '80kV',   5,     5],
               ['detunedMV','80kV',   9,     1] ]


# coords of ROIs for SNR measurements, dx, dy, x0, y0, x02, y02
# x0, y0 gives signal and x02, y02 gives background
roi_dict = { 'lung'  :   [5,  5,  291, 271, 291, 280],
             'pelvis':   [20, 20, 246, 242, 278, 284],
             'prost' :   [15, 15, 248, 271, 274, 280] }
roi_dict['pelvis_metal'] = roi_dict['pelvis']
roi_dict['lung_metal'] = roi_dict['lung']


# monoenergies for VMIs
E0s = [60, 200]


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
    mask[y0:y0+dy, x0:x0+dy] = 1
    ROI = M[mask]
    u, v = np.mean(ROI), np.var(ROI)
    if ax is not None:
        xvals = [x0+dx, x0, x0, x0+dx, x0+dx]
        yvals = [y0, y0, y0+dy, y0+dy, y0]
        ax.plot(xvals, yvals, 'r-', lw=0.5)
    if give_roi:
        return ROI
    return u,v

def make_cbar(axi, m, label, 
              cbar_fsz=8, cbar_pad=0.01, cbar_lab_pad=-5):
    cbar = fig.colorbar(m, ax=axi, pad=cbar_pad)
    cbar.ax.tick_params(labelsize=cbar_fsz) 
    cbar.set_label(label, labelpad=cbar_lab_pad)
   
        
def get_ssmat(geo_id, phantom, spec, D, ramp=1, N_matrix=512, N_cropped=50):
    cols = ct.N_channels
    proj = ct.N_proj
    sdir = rootd+f'{phantom}/{spec}/{geo_id}/{int(1000*D)}uGy/'
    recon_fname = sdir+f'recon512_ramp{int(100*ramp)}.npy' 
    if ramp==1:
        M = np.fromfile(sdir+'recon512_000.npy', dtype=np.float32).reshape([512,512]).T
    elif ramp!=1 and os.path.exists(recon_fname):
        M = np.fromfile(recon_fname, dtype=np.float32).reshape([512,512])
    else:
        # recon with new ramp
        sino =  np.fromfile(sdir+'sino_log_000.npy', dtype=np.float64).reshape([proj, cols])  # GLJ TYPE
    
        sino_filtered = pre_process(sino, ct, ramp_percent=1.0)
 
        ji_coord, r_M, theta_M, gamma_target_M, L2_M = get_recon_coords(
            N_matrix, N_cropped, ct.N_proj, ct.dtheta_proj, ct.SID)
        
        M = do_recon_gpu(sino_filtered, gamma_target_M, L2_M, ct.gammas, ct.dtheta_proj)
        M = M.T
        # plot_recon(recon, title=f'mat {i+1}, {run_id}', vmi=0, vma=[1.2, 2.2][i])
        
        # convert to HU 
        u_w = u_water_dict[spec]
        u_a = u_air_dict[spec]
        M = 1000*(M-u_w)/(u_w-u_a)
        M.astype(np.float32).tofile(recon_fname)  #float32
        
    return M      
        
def get_Mmat(geo_id, phantom, spec1, spec2, D1, D2, ramp=1, ct=ct, N_matrix=512, N_cropped=50):
    cols = ct.N_channels
    proj = ct.N_proj
    # basis materials
    md_dir = rootd+f'matdecomp/{geo_id}/{phantom}/{spec1}_{spec2}_{int(1000*D1)}uGy_{int(1000*D2)}uGy/'
    print(md_dir)
    if ramp==1:
        M_m1 = np.fromfile(md_dir+'recon_50iters_mat1.npy', dtype=np.float32).reshape([512,512])
        M_m2 = np.fromfile(md_dir+'recon_50iters_mat2.npy', dtype=np.float32).reshape([512,512])
    elif ramp!=1 and os.path.exists(md_dir+f'recon_ramp{int(100*ramp)}_mat1.npy'):
        M_m1 = np.fromfile(md_dir+f'recon_ramp{int(100*ramp)}_mat1.npy', dtype=np.float32).reshape([512,512])
        M_m2 = np.fromfile(md_dir+f'recon_ramp{int(100*ramp)}_mat2.npy', dtype=np.float32).reshape([512,512])
    else:
        # recon with new ramp
        print('new recon', md_dir)
        sino_m1 =  np.fromfile(md_dir+'sino_mat1.npy', dtype=np.float32).reshape([proj, cols])
        sino_m2 =  np.fromfile(md_dir+'sino_mat2.npy', dtype=np.float32).reshape([proj, cols])
        sinos = [sino_m1, sino_m2]
        recons = []
        for i in range(2):
            # # recon  
            print(ramp)
            sino_filtered = pre_process(sinos[i], ct, ramp)
            ji_coord, r_M, theta_M, gamma_target_M, L2_M = get_recon_coords(
                N_matrix, N_cropped, ct.N_proj, ct.dtheta_proj, ct.SID)
            
            recon = do_recon_gpu(sino_filtered, gamma_target_M, L2_M, ct.gammas, ct.dtheta_proj)
            recon = recon.T
            recons.append(recon)
            recon_fname = md_dir+f'recon_ramp{int(100*ramp)}_mat{i+1}.npy' 
            recon.astype(np.float32).tofile(recon_fname)  #float32
        M_m1, M_m2 = recons

    return M_m1, M_m2
                

    
#%% Cell 
phantom, WL, WW, N_cropped =  'pelvis',  50,  400, 500
   
filename_phantom = f'inputs/phantom/xcat_{phantom}_uint8_512_512_1_1mm.bin'
filename_matcomp = 'inputs/phantom/xcat_elemental_comps.txt'
xcat = Phantom(filename_phantom, phantom, filename_matcomp, 512, 512, 1, ind=0) 
 
_, _, x0, y0, x02, y02 = np.array(roi_dict[phantom]) - (N_matrix - N_cropped)//2
dx, dy, _, _, _, _ = roi_dict[phantom]
hu_kwargs = {'cmap':'gray', 'vmin':WL-WW/2, 'vmax':WL+WW/2}

# generate ground truth + scale/shift to register
M = xcat.M_mono(80)  # ground truth
X = np.linspace(0, 512., 512)
Xn = np.linspace(0, 512., 524)
f = RectBivariateSpline(X, X, M)  # this is somewhat imperfect
dgtx, dgty = 10, 10
M_80k = f(Xn, Xn)[dgty+r0-N_cropped//2:dgty+r0+N_cropped//2, dgtx+r0-N_cropped//2:dgtx+r0+N_cropped//2]

M = xcat.M_mono(1000)  # ground truth
X = np.linspace(0, 512., 512)
Xn = np.linspace(0, 512., 524)
f = RectBivariateSpline(X, X, M)  # this is somewhat imperfect
dgtx, dgty = 10, 10
M_1M = f(Xn, Xn)[dgty+r0-N_cropped//2:dgty+r0+N_cropped//2, dgtx+r0-N_cropped//2:dgtx+r0+N_cropped//2]

fig,ax = plt.subplots(1,2,figsize=[8,4], dpi=300)
ax[0].imshow(M_80k, **hu_kwargs)
ax[1].imshow(M_1M, **hu_kwargs)
fig.tight_layout()
plt.show()


for spec1, spec2, D1, D2, ls, sid in [
        #['140kV',     '80kV', 5, 5,  'bs-', '140-80kV'],
        ['detunedMV', '120kV', 9, 1,  'ro-', 'MV-80kV'],
        ]:
    print()
    print(f'{phantom} / {spec1}-{spec2}',)
    print(f'{"":12}        SNR     Noise      RMSE',)

    get_out_dir_se(ct, phantom, spec, D1)
    s1_dir = rootd+f'{phantom}/{spec1}/{geo_id}/{int(1000*D1)}uGy/'
    s2_dir = rootd+f'{phantom}/{spec2}/{geo_id}/{int(1000*D2)}uGy/'
    M_s1 = np.fromfile(s1_dir+'recon512_000.npy', dtype=np.float32).reshape([512,512]).T
    M_s2 = np.fromfile(s2_dir+'recon512_000.npy', dtype=np.float32).reshape([512,512]).T
    
    # basis materials
    M_m1, M_m2 = get_Mmat(geo_id, phantom, spec1, spec2, D1, D2)
    
    # crop
    M_s1 = M_s1[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
    M_s2 = M_s2[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
    M_m1 = M_m1[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
    M_m2 = M_m2[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]

    vmi_80k = make_vmi(80, M_m1, M_m2)
    vmi_1M = make_vmi(1000, M_m1, M_m2)


#%%

roi_80k = measure_roi(M_80k, [x0, y0, dx, dy], give_roi=True)   # signal
roi_vmi_80k = measure_roi(vmi_80k, [x0, y0, dx, dy], give_roi=True)   # signal
u_1M, _ = measure_roi(M_1M, [x0, y0, dx, dy])   # signal
#u_1M, _ = measure_roi(M, [x02, y02, dx, dy], ax=ax[i]) # background
       
print(np.mean(roi_80k), np.std(roi_80k))
print(np.mean(roi_vmi_80k), np.std(roi_vmi_80k))

# rmse
print(np.sqrt(np.mean((M_80k-vmi_80k)**2)))

#%% Cell 2
test_rois = True
if test_rois:
    #fig, ax = plt.subplots(1,3, dpi=300, figsize=[8,2] )
    fig, ax = plt.subplots(1,2, dpi=300, figsize=[6,1.8] )
    if len(ax)==1:
        ax = [ax]
    for i, [phantom, WL, WW, N_cropped] in enumerate([
                #['lung', -500, 1000, 500 ],
                ['pelvis',  100,  500, 500],
                ['pelvis_metal',  100,  500, 500],
                #['prost',   50,  400, 500],
                ]):
        print(phantom)
        
        filename_phantom = f'inputs/phantom/xcat_{phantom}_uint8_512_512_1_1mm.bin'
        filename_matcomp = 'inputs/phantom/xcat_elemental_comps.txt'
        xcat = Phantom(filename_phantom, phantom, filename_matcomp, 512, 512, 1, ind=0) 
        
            
        _, _, x0, y0, x02, y02 = np.array(roi_dict[phantom]) - (N_matrix - N_cropped)//2
        dx, dy, _, _, _, _ = roi_dict[phantom]
        hu_kwargs = {'cmap':'gray', 'vmin':WL-WW/2, 'vmax':WL+WW/2}
        
        # scale/shift the ROIs register
        M = xcat.M_mono(80)  # ground truth
        scale = 524/512
        dgtx, dgty = int(scale*10), int(scale*10)
        
        m = ax[i].imshow(M, **hu_kwargs)
        #make_cbar(ax[i], m, 'HU')        
        Y0 = 140
        ax[i].text(5,Y0+5,f'W/L = {WW}/{WL}',color='w', ha='left', va='top')
        measure_roi(M, [x0+dgtx,  y0+dgty,  int(dx*scale), int(dy*scale)], ax=ax[i])   # signal
        measure_roi(M, [x02+dgtx, y02+dgty, int(dx*scale), int(dy*scale)], ax=ax[i]) # background
                
        ax[i].axis('off')
        ax[i].grid('off')
        ax[i].set_title(phantom.replace('_metal', ' with metal'))
        ax[i].set_ylim(400, Y0)
 
    fig.tight_layout()
    plt.savefig(figdir+'phantom_rois.pdf')
    plt.show()
    
#%%

phantlist = [    
            ['pelvis',       50, 400, 200,  np.arange(50, 155, 10)],
            ['pelvis_metal', 50, 400, 200,  np.arange(100, 305, 20)],
            ]
height =3.5
legwidth = 0.3  # all plots
fig, ax = plt.subplots(1,len(phantlist), dpi=300, 
                               figsize=[len(phantlist)*height + legwidth, height])

ramp = 1.0
phantnames = f'_{int(100*ramp)}ramp_pelvis_pelvis_metal'
for i, [phantom, WL, WW, N_cropped, Evals] in enumerate(phantlist):
    _, _, x0, y0, x02, y02 = np.array(roi_dict[phantom]) - (N_matrix - N_cropped)//2
    dx, dy, _, _, _, _ = roi_dict[phantom]
    hu_kwargs = {'cmap':'gray', 'vmin':WL-WW/2, 'vmax':WL+WW/2}
    ax[i].set_title(phantom.replace('_',' ').replace('metal', 'with metal'))
    filename_phantom = f'inputs/phantom/xcat_{phantom}_uint8_512_512_1_1mm.bin'
    filename_matcomp = 'inputs/phantom/xcat_elemental_comps.txt'
    xcat = Phantom(filename_phantom, phantom, filename_matcomp, 512, 512, 1, ind=0) 
    
    # Evals = [80]  # for testing! 
    
    # dual specs
    for spec1, spec2, D1, D2, ls, sid in [
            ['140kV',     '80kV', 5, 5,  'bs-', '140-80kV'],
            ['detunedMV', '80kV', 9, 1,  'ro-', 'MV-80kV'],
            ]:
        print()
        print(f'{phantom} / {spec1}-{spec2}',)
        
        # basis materials
        md_dir = rootd+f'matdecomp/{geo_id}/{phantom}/{spec1}_{spec2}_{int(1000*D1)}uGy_{int(1000*D2)}uGy/'
        M_m1, M_m2 = get_Mmat(geo_id, phantom, spec1, spec2, D1, D2, ramp=ramp)

        rmses = []
        rmses_ss = [[],[],[],[]]
            
        for E0 in Evals:
            
            # generate vmi 
            vmi = make_vmi(E0, M_m1, M_m2)
            
            # at E0, generate ground truth + scale/shift to register
            M = xcat.M_mono(E0)  # ground truth
            X = np.linspace(0, 512., 512)
            Xn = np.linspace(0, 512., 524)
            f = RectBivariateSpline(X, X, M)  # this is somewhat imperfect
            dgtx, dgty = 10, 10
            M_gt = f(Xn, Xn)[dgty:dgty+512,dgtx:dgtx+512]
                
            mask = np.ones([512,512], dtype=bool)
            #mask[235:285, 135:195] = 0
            M_gt = M_gt[mask]

            vmi = vmi[mask]
            rmse = np.sqrt( np.mean( (vmi-M_gt)**2 ) )
            rmses.append(rmse)
            #print(E0, rmse)
            
            # single specs
            for ii, [spec, D, _] in enumerate( [
                ['80kV', 10, '--'], 
                ['120kV',10,':'], 
                ['140kV',10,'-'],
                ]):
                col='k'
                #M = get_ssmat(geo_id, phantom, spec, D, ramp=ramp)            
                #M = M[mask]
                #rmse = np.sqrt( np.mean( (M-M_gt)**2 ) )
                
                #rmses_ss[ii].append(rmse)
    
        print(sid, np.min(rmses))
        ax[i].plot(Evals, rmses, ls, markerfacecolor="None", label=sid)
        for ii, [spec, D, ls] in enumerate( [
                ['80kV', 10, '--'], 
                ['120kV',10,':'], 
                ['140kV',10,'-'],
                ]):
            tlw=1
            #ax[i].plot(Evals, rmses_ss[ii], lw=tlw, color=col, ls=ls, label=spec)
            
ax[0].set_ylabel('RMSE [HU]')
for axi in ax:
    #axi.legend(fontsize=6, loc='best', framealpha=1.0)
    axi.set_xlabel('VMI energy [keV]')
    x0,x1 = axi.get_xlim()
    y0,y1 = axi.get_ylim()
    axi.set_aspect(0.9*(x1-x0)/(y1-y0))
    
legend_elements = [
    plt.Line2D([0], [0], color='r', marker='o', markerfacecolor='None', label='MV-80kV'),
    plt.Line2D([0], [0], color='b', marker='s', markerfacecolor='None', label='140-80kV'),
    #plt.Line2D([0], [0], color='k', ls='--',  label='80kV'),
    #plt.Line2D([0], [0], color='k', ls=':',  label='120kV'),
    #plt.Line2D([0], [0], color='k', ls='-',  label='140kV'),
               ]
fig.tight_layout(pad=1.1)
fig.legend(handles=legend_elements, loc='center right')
plt.subplots_adjust(right = 0.86)# len(phantlist)*height/(len(phantlist)*height + legwidth))   
plt.savefig(figdir+f'rmse{phantnames}.pdf')
plt.show()

    
        
#%%
        
measure_vmis = True
do_noise = False
do_snr = True
do_contrast = False
ramp = .8
phantnames = f'_{int(100*ramp)}ramp_pelvis_pelvis_metal'
phantlist = [
            #['lung', -500, 1000, 500 ],
            #['lung_metal', -500, 1000, 500 ],
            ['pelvis',       50, 400, 200,  np.arange(40, 205, 10)],
            ['pelvis_metal', 50, 400, 200,  np.arange(40, 365, 20)],
            #['pelvis',       50, 400, 200,  np.arange(40, 65, 20)],
            #['pelvis_metal', 50, 400, 200,  np.arange(40, 65, 20)],
            #['prost', 50, 400, 200],
            ]
if measure_vmis:
    if do_snr or do_noise:
        height =3.5
        legwidth = 0.3  # all plots
        fig, ax = plt.subplots(1,len(phantlist), dpi=300, 
                               figsize=[len(phantlist)*height + legwidth, height])
    
    for i, [phantom, WL, WW, N_cropped, Evals] in enumerate(phantlist):
        _, _, x0, y0, x02, y02 = np.array(roi_dict[phantom]) - (N_matrix - N_cropped)//2
        dx, dy, _, _, _, _ = roi_dict[phantom]
        hu_kwargs = {'cmap':'gray', 'vmin':WL-WW/2, 'vmax':WL+WW/2}
        
        if not do_snr and not do_noise: # for each phantom, measure SNR and noise both
            fig, ax = plt.subplots(1,2, dpi=300, figsize=[7,3])
        
        # single specs
        for spec, D, ls in  [
                #['detunedMV', 10, '-'],
                ['80kV', 10, '--'], 
                ['120kV',10,':'], 
                ['140kV',10,'-'],
                ]:
            col='k'
            M = get_ssmat(geo_id, phantom, spec, D, ramp=ramp)            
            M = M[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]

            u1, v1 = measure_roi(M, [x0, y0, dx, dy])#, ax=ax[i])   # signal
            u2, v2 = measure_roi(M, [x02, y02, dx, dy])#, ax=ax[i]) # background
                                    
            noise = np.sqrt(v1)
            snr = (u1-u2)/np.sqrt(v1 + v2)
            if do_contrast:
                snr = np.abs(u1-u2)  # ignore noise
            print(phantom, spec, snr)

            
            tlw=1.5
            if not do_snr and not do_noise:
                ax[0].axhline(snr, lw=tlw, color=col, ls=ls, label=spec)
                ax[1].axhline(noise,lw=tlw,  color=col, ls=ls, label=spec)
            elif do_snr:
                ax[i].axhline(snr, lw=tlw, color=col, ls=ls, label=spec)
            elif do_noise:
                ax[i].axhline(noise, lw=tlw, color=col, ls=ls, label=spec)

            #print(f'{spec:12} - {snr:8.3f}, {noise:8.3f}',)

        # dual specs
        for spec1, spec2, D1, D2, ls, sid in [
                ['140kV',     '80kV', 5, 5,  'bs-', '140-80kV'],
                ['detunedMV', '80kV', 9, 1,  'ro-', 'MV-80kV'],
                ]:
            print()
            print(f'{phantom} / {spec1}-{spec2}',)
            print(f'{"":12}        SNR     Noise      RMSE',)

            # basis materials
            md_dir = rootd+f'matdecomp/{geo_id}/{phantom}/{spec1}_{spec2}_{int(1000*D1)}uGy_{int(1000*D2)}uGy/'
            M_m1, M_m2 = get_Mmat(geo_id, phantom, spec1, spec2, D1, D2, ramp=ramp)

            # crop
            M_m1 = M_m1[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
            M_m2 = M_m2[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
        
            rmses = []
            noises = []
            snrs = []
            #Evals = np.arange(40, 205, 10)
            #np.arange(40, 100, 10)
            #Evals = [50, 60, 70, 80, 120, 190, 200, 210, 220, 300]
            for E0 in Evals:
                vmi = make_vmi(E0, M_m1, M_m2)
 
                # measurements
                u1, v1 = measure_roi(vmi, [x0, y0, dx, dy])#, ax=ax[i])   # signal
                u2, v2 = measure_roi(vmi, [x02, y02, dx, dy])#, ax=ax[i]) # background
                                        
                noise = np.sqrt(v1)
                snr = (u1-u2)/np.sqrt(v1 + v2)
                if do_contrast:
                    snr = np.abs(u1-u2)  # ignore noise
                    
                noises.append(noise)
                snrs.append(snr)
                
                #print(f'{E0:8} keV - {snr:8.3f}, {noise:8.3f}',)
            
            #sid = 'VMI' #f'{spec1}-{spec2}'
            if not do_snr and not do_noise:
                ax[0].plot(Evals, snrs, ls, markerfacecolor="None", label=sid)
                ax[1].plot(Evals, noises, ls, markerfacecolor="None", label=sid)
            elif do_snr:
                ax[i].plot(Evals, snrs, ls, markerfacecolor="None", label=sid)
            elif do_noise:
                ax[i].plot(Evals, noises, ls, markerfacecolor="None", label=sid)

        if do_snr or do_noise:
            ax[i].set_title(phantom.replace('_',' ').replace('metal', 'with metal'))
        if not do_snr and not do_noise:
            ax[0].set_title('SNR')
            ax[1].set_title('Noise [HU]')
        elif do_snr:
            ax[0].set_ylabel('SNR')
            if do_contrast:
                ax[0].set_ylabel('Contrast [$\\Delta$HU]')
        elif do_noise:
            ax[0].set_ylabel('Noise [HU]')

    for axi in ax:
        #axi.legend(fontsize=6, loc='best', framealpha=1.0)
        axi.set_xlabel('VMI energy [keV]')
        x0,x1 = axi.get_xlim()
        y0,y1 = axi.get_ylim()
        #axi.set_aspect(0.7*(x1-x0)/(y1-y0))
        axi.set_aspect(0.9*(x1-x0)/(y1-y0))

    legend_elements = [
        plt.Line2D([0], [0], color='r', marker='o', markerfacecolor='None', label='MV-80kV'),
        plt.Line2D([0], [0], color='b', marker='s', markerfacecolor='None', label='140-80kV'),
        plt.Line2D([0], [0], color='k', ls='--',  label='80kV'),
        plt.Line2D([0], [0], color='k', ls=':',  label='120kV'),
        plt.Line2D([0], [0], color='k', ls='-',  label='140kV'),
                   ]
    fig.tight_layout(pad=1.1)
    #fig.tight_layout(pad=0.4)
    fig.legend(handles=legend_elements, loc='center right')
    plt.subplots_adjust(right=0.86)   # len 2


    if not do_snr and not do_noise:
        plt.savefig(figdir+f'snr_noise{phantnames}.pdf')
    elif do_snr:
        plt.savefig(figdir+f'snr{phantnames}.pdf')
    elif do_noise:
        plt.savefig(figdir+f'noise{phantnames}.pdf')
    plt.show()

    
    
plot_vmis = False
crop = True
if plot_vmis:
    
    for E0 in [200, 300, 400]: #[60, 100, 200]:
        for spec1, spec2, D1, D2, ls, sid in [
                ['140kV',     '80kV', 1,   1,  'bs-', '140kV-80kV VMI'],
                ['detunedMV', '80kV', 10 , 1,  'ro-', 'MV-80kV VMI'],
                ]:
            _, _, x0, y0, x02, y02 = np.array(roi_dict[phantom]) - (N_matrix - N_cropped)//2
            dx, dy, _, _, _, _ = roi_dict[phantom]
            
            #fig, ax = plt.subplots(1,5,dpi=300, figsize=[10,2.5])
            fig, ax = plt.subplots(1,2,dpi=300, figsize=[8,3])
            fig.suptitle(f'{E0} keV, {spec1}-{spec2}', y=.95)
            ax[0].set_ylabel(f'{E0} keV, {spec1}-{spec2}')
            fig.subplots_adjust(top=0.8)
            
            for i, [phantom, WL, WW, N_cropped] in enumerate([
                        #['lung',       -500, 200,  200 ],
                        #['lung_metal', -500, 200,  200 ],
                        ['pelvis',       50, 400,  200],
                        ['pelvis_metal', 50, 400 , 200 ],
                        #['prost',        50, 200,  200],
                        ]):
        
                _, _, x0, y0, x02, y02 = np.array(roi_dict[phantom]) - (N_matrix - N_cropped)//2
                dx, dy, _, _, _, _ = roi_dict[phantom]
                hu_kwargs = {'cmap':'gray', 'vmin':WL-WW/2, 'vmax':WL+WW/2}
     
                s1_dir = rootd+f'{phantom}/{spec1}/{geo_id}/{int(1000*D1)}uGy/'
                s2_dir = rootd+f'{phantom}/{spec2}/{geo_id}/{int(1000*D2)}uGy/'
                M_s1 = np.fromfile(s1_dir+'recon512_000.npy', dtype=np.float32).reshape([512,512]).T
                M_s2 = np.fromfile(s2_dir+'recon512_000.npy', dtype=np.float32).reshape([512,512]).T
                
                # basis materials
                md_dir = rootd+f'matdecomp/{geo_id}/{phantom}/{spec1}_{spec2}_{int(1000*D1)}uGy_{int(1000*D2)}uGy/'
                M_m1, M_m2 = get_Mmat(geo_id, phantom, spec1, spec2, D1, D2)

                # crop
                if not crop: 
                    N_cropped = 400
                if True:
                    M_s1 = M_s1[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
                    M_s2 = M_s2[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
                    M_m1 = M_m1[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
                    M_m2 = M_m2[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
                
                vmi = make_vmi(E0, M_m1, M_m2)
                m = ax[i].imshow(vmi, **hu_kwargs)
                u1, v1 = measure_roi(vmi, [x0, y0, dx, dy], ax=ax[i])   # signal
                u2, v2 = measure_roi(vmi, [x02, y02, dx, dy], ax=ax[i]) # background
                  
                # make_cbar(ax[i], m, 'HU')        
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].text(12,12,f'W/L = {WW}/{WL}',color='w', ha='left', va='top', bbox=dict(facecolor='k'))
                ax[i].set_title(phantom.replace('_', ' ').replace('prost', 'prostate'))
            fig.tight_layout(pad=0)
            fname = f'vmi_{spec1}_{spec2}_{E0}keV.pdf'
            if crop:
                fname = fname.replace('.pdf', '_crop.pdf')
            plt.savefig(figdir+fname)
            plt.show()
        
        
    
#%%

E0s = [80, 300]
ramp = 0.8

for phantom, WL, WW, N_cropped in [
            ['pelvis', 50, 500, 380],
            ['pelvis_metal', 50, 500 , 380 ],
            ]:
    hu_kwargs = {'cmap':'gray', 'vmin':WL-WW/2, 'vmax':WL+WW/2}
    mat1_kwargs = {'cmap':'gray', 'vmin':0, 'vmax':1.2}  # turbo
    mat2_kwargs = {'cmap':'gray', 'vmin':0, 'vmax':2.2}

    for spec1, spec2, D1, D2 in spec_pairs:

        
        geo_id = '60cm_100cm_47fan_1200view_800col'
        M_s1 = get_ssmat(geo_id, phantom, spec1, D1, ramp=ramp)            
        M_s2 = get_ssmat(geo_id, phantom, spec2, D2, ramp=ramp)      
        M_m1, M_m2 = get_Mmat(geo_id, phantom, spec1, spec2, D1, D2, ramp=ramp)

        M_s1 = M_s1[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
        M_s2 = M_s2[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
        M_m1 = M_m1[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
        M_m2 = M_m2[r0-N_cropped//2:r0+N_cropped//2, r0-N_cropped//2:r0+N_cropped//2]
            
        vmi1 = make_vmi(E0s[0], M_m1, M_m2)
        vmi2 = make_vmi(E0s[1], M_m1, M_m2)
        
 
        fig, ax = plt.subplots(3,2, dpi=300, figsize=[6.3,8])

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
        fname = f'{phantom}_{spec1}_{spec2}_{int(D1)}mGy_{int(D2)}mGy_ramp{int(100*ramp)}_{int(E0s[0])}keV_{int(E0s[1])}keV.pdf'
        plt.savefig(figdir+fname)
        plt.show()




