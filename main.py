#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 13:30:24 2022

@author: gjadick

Script for:
    (1) Generating single-energy CT sinograms using Siddons raytracing.
    (2) Reconstructing CT images using fan-beam filtered back-projection.
    (3) Generating dual-energy CT basis material sinograms using Gauss-Newton 
        decomposition and reconstructing basis material images using FFBP.

"""

import sys
sys.path.append('xtomosim')  # for xtomosim

from xtomosim.system import read_parameter_file, xRaySpectrum
from xtomosim.forward_project import get_sino
from xtomosim.back_project import get_recon
from matdecomp import get_basismat_sinos

import os
import shutil
from time import time
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    'figure.dpi': 300,
    'font.size':10,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': True,
    'axes.titlesize':10,
    'axes.labelsize':8,
    'axes.linewidth': .5,
    'xtick.top': True, 
    'ytick.right': True, 
    'xtick.direction': 'in', 
    'ytick.direction': 'in',
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'axes.grid' : False, 
    'grid.color': 'lightgray',
    'grid.linestyle': ':',
    'legend.fontsize':8,
    'lines.linewidth':1,
    'image.cmap':'gray',
    'image.aspect':'auto'
    })


def ax_imshow(fig, ax, img, colorbar=True, title='', kw={}):
    """Convenience function for showing images with colorbar."""
    ax.set_title(title)
    m = ax.imshow(img, **kw)
    if colorbar:
        fig.colorbar(m, ax=ax)
        
        
def load_spectrum(spec_id, dose):
    """Convenience function for loading DECT spectra and scaling dose."""
    fname = f'./input/spectrum/{spec_id}_1mGy_float32.bin'
    spec = xRaySpectrum(fname, spec_id)
    spec.rescale_counts(ct.A_iso * dose / ct.N_proj)  
    return spec




if __name__ == '__main__':
    
    
    ###########################################################################
    ### INPUTS
    
    param_file = './input/params.txt'  # only requires geometry and phantom
    main_output_dir = './output/'
    show_imgs = True
    
    ### END OF INPUTS
    ###########################################################################
    
    
    # Unpack scanner geometry and phantom parameters.
    params = read_parameter_file(param_file)[0]
    run_id, do_forward_projection, do_back_projection = params[:3]
    ct, phantom, _ = params[3:6]  # ignore spectrum, assign in DECT main loop below
    if do_back_projection:
        N_matrix, FOV, ramp = params[6:9]
            
    out_dir = os.path.join(main_output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(param_file, os.path.join(out_dir, 'params.txt'))
    
    # Main loop over all the dual energy spectral combinations.
    for spec_id1, spec_id2, D1, D2 in [['detunedMV', '80kV', 9, 1],
                                       ['140kV',     '80kV', 5, 5]]:
        t0 = time()
        dect_specs = []
        dect_sinos = []
        
        # 1 : SINGLE ENERGY
        for spec_id, dose in [[spec_id1, D1], [spec_id2, D2]]:

            sub_dir = os.path.join(out_dir, f'{spec_id}_{int(dose*1000):04}uGy/') 
            os.makedirs(sub_dir, exist_ok=True)
            print(f'\n*** {sub_dir} ***')
            
            spec = load_spectrum(spec_id, dose)
            dect_specs.append(spec)
            
            # 1a : Forward project for one spectrum.
            print('Forward projecting!')
            sino_raw, sino_log = get_sino(ct, phantom, spec)   
            sino_raw.astype(np.float32).tofile(sub_dir+'sino_raw_float32.bin')
            sino_log.astype(np.float32).tofile(sub_dir+'sino_log_float32.bin')
            if show_imgs:
                fig, ax = plt.subplots(1, 2, figsize=[7,3])
                ax_imshow(fig, ax[0], sino_raw, title='Raw line integrals')
                ax_imshow(fig, ax[1], sino_log, title='Log sinogram')
                fig.tight_layout()
                plt.show()
            dect_sinos.append(sino_raw)

            # 1b : Reconstruct the one image.
            if do_back_projection:
                print('Back projecting!')
                recon_raw, recon_HU = get_recon(sino_log, ct, spec, N_matrix, FOV, ramp) 
                recon_raw.astype(np.float32).tofile(sub_dir+'recon_raw_float32.bin')
                recon_HU.astype(np.float32).tofile(sub_dir+'recon_HU_float32.bin')
                if show_imgs:
                    fig, ax = plt.subplots(1, 2, figsize=[7,3])
                    ax_imshow(fig, ax[0], recon_raw, title='Raw reconstruction [cm$^{-1}$]')
                    ax_imshow(fig, ax[1], recon_HU, title='Hounsfield Units')
                    fig.tight_layout()
                    plt.show()

        # 2 : DUAL ENERGY
        spec1, spec2 = dect_specs
        sino1, sino2 = dect_sinos
        sub_dir = os.path.join(out_dir, f'matdecomp_{spec_id1}_{spec_id2}_{int(D1*1000):04}uGy_{int(D2*1000):04}uGy/') 
        os.makedirs(sub_dir, exist_ok=True)
        print(f'\n*** {sub_dir} ***')

        # 2a : Apply Gauss-Newton material decomposition to get basis material sinograms.
        print('Decomposing into basis material sinograms!')
        matsino1, matsino2 = get_basismat_sinos(ct, sino1, sino2, spec1, spec2, n_iters=50)
        matsino1.astype(np.float32).tofile(sub_dir+'mat1_sino_float32.bin')
        matsino2.astype(np.float32).tofile(sub_dir+'mat2_sino_float32.bin')
        if show_imgs:
            fig, ax = plt.subplots(1, 2, dpi=300, figsize=[7,3])
            ax_imshow(fig, ax[0], matsino1, title='Basis material 1')
            ax_imshow(fig, ax[1], matsino2, title='Basis material 2')
            fig.tight_layout()
            plt.show()
            
        # 2b : Reconstruct into basis material images.
        if do_back_projection:
            print('Back projecting basis material sinograms!')
            recons = []
            for i, matsino in enumerate([matsino1, matsino2]):
                recon_raw, _ = get_recon(matsino, ct, spec1, N_matrix, FOV, ramp)  # spec is filler
                recon_raw.astype(np.float32).tofile(sub_dir+f'mat{i+1}_recon_float32.bin')
                recons.append(recon_raw)
            if show_imgs:
                fig, ax = plt.subplots(1, 2, figsize=[7,3])
                ax_imshow(fig, ax[0], recons[0], title='Basis material 1')
                ax_imshow(fig, ax[1], recons[1], title='Basis material 2')
                fig.tight_layout()
                plt.show()

        print(f'matdecomp finished for {spec_id1}-{spec_id2} : t={time() - t0:.2f}s') 



