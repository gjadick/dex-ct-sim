#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:16:53 2023

@author: giavanna

Helpful for loading things.

"""

import os
import numpy as np
import cupy as cp
from forward_project import raytrace_fanbeam, detect_transmitted_sino
from back_project import pre_process, get_recon_coords, do_recon
from matdecomp import do_matdecomp_gn


def get_out_dir_se(ct, phantom, spec, dose):
    '''
    Returns output directory name for single-energy run.
    Unique for each ct geometry, phantom, spectrum, and dose level.
    '''
    return f'output/{ct.geo_id}/{phantom.name}/{spec.name}/{int(dose*1000):04}uGy/'     


def get_out_dir_de(ct, phantom, spec1, spec2, dose1, dose2):
    '''
    Returns the output directory name for dual-energy mat decomp run.
    Unique for each ct geometry, phantom, and DE spectral combination.    
    '''
    return f'output/{ct.geo_id}/{phantom.name}/matdecomp_{spec1.name}_{spec2.name}/{int(dose1*1000):04}uGy_{int(dose2*1000):04}uGy/'     


def get_sino(out_dir, ct, phantom, spec):
    '''
    Forward projects through the phantom using a given ct geometry and
    input polychromatic spectrum.

    Parameters
    ----------
    out_dir : str
        path to directory for saving output files
    ct : ScannerGeometry
        collection of parameters defining the CT acquisition geometry
    phantom : Phantom
        object through which to raytrace.
    spec : Spectrum
        polychromatic x-ray spectrum. Magnitude should be scaled to target dose.

    Returns
    -------
    sino_raw : 2D numpy array (float32)
        The raw sinogram in counts.
    sino_log : 2D numpy array (float32)
        The logged sinogram normalized to the zero sinogram, i.e. -ln(I/I0)
    '''
    # make output directory
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # forward project
    sino_raw = raytrace_fanbeam(ct, phantom, spec)
    
    # log the data
    #sino0 = np.ones([ct.N_proj, ct.N_channels])*detect_transmitted_sino(spec.E, spec.I0, cp.ones([1,1,1]), ct, noise=False).get()
    sino0 = detect_transmitted_sino(spec.E, spec.I0, cp.ones([1,1,1]), ct, noise=False).get()
    sino_log = np.log(sino0/sino_raw)

    # save sinos
    sino_raw.astype(np.float32).tofile(out_dir+'sino_raw.bin')
    sino_log.astype(np.float32).tofile(out_dir+'sino_log.bin')

    return sino_raw, sino_log


def get_recon(out_dir, sino_log, ct, N_matrix, FOV, ramp, HU=False, spec=None, name=''):
    '''
    Reconstruct a CT sinogram into a cross-sectional image.

    Parameters
    ----------
    out_dir : str
        path to directory for saving output files
    sino_log : 2D numpy array (float32), shape [N_proj, N_channels]
        The input sinogram. For normal CT recon, this should be the log data.
        For a basis material sinogram, this should be the density line integrals.
    ct : ScannerGeometry
        collection of parameters defining the CT acquisition geometry
    N_matrix : int
        Number of pixels in the reconstructed matrix, shape [N_matrix, N_matrix]
    FOV : float
        Size of field-of-view to reconstruct, units cm.
    ramp : float, 0 to 1
        Cutoff fraction of Nyquist frequency for the recon filter.
    HU : bool, optional
        Whether to convert pixel units to Hounsfield Units. 
        If True, must assign a Spectrum (spec parameter)
        The default is False.
    spec: Spectrum, optional
        The polychromatic x-ray spectrum for the acquisition. 
        Necessary for getting the effective linear attenuation coefficients
        for HU conversion.
        The default is None.
    name : str, optional
        String to tag onto the start of the output file name. The default is ''.

    Returns
    -------
    recon : 2D numpy array, shape [N_matrix, N_matrix].
        The reconstructed image.
    '''
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    sino_filtered = pre_process(sino_log, ct, ramp)
    r_matrix, theta_matrix = get_recon_coords(N_matrix, FOV)
    recon = do_recon(sino_filtered, r_matrix, theta_matrix, ct.SID, ct.dgamma_channel, ct.dtheta_proj)
    
    # convert to HU
    if HU:
        if spec is None:
            print('HU conversion failed, spec must be assigned to get mu values')
        else:
            recon = 1000*(recon - spec.u_water)/(spec.u_water - spec.u_air)
    
    # save recons
    recon_fname = out_dir+f'{name}recon_{N_matrix}_{int(FOV):02}cm_{int(ramp*100):03}ramp.bin' 
    recon.astype(np.float32).tofile(recon_fname)  
    
    return recon


def get_matdecomp(ct, phantom, spec1, spec2, D1, D2, N_matrix, FOV, ramp, n_iters=30, mask_thresh=0.95):
    '''
    For a DE-CT acquisition, make basis material sinograms using a
    Gauss-Newton algorithm and reconstruct into basis material images.The 
    basis materials are ICRU tissue (mat1) and bone (mat2). If desired, 
    these can be changed in matdecomp.do_matdecomp_gn().
    
    Parameters
    ----------
    ct : ScannerGeometry
        collection of parameters defining the CT acquisition geometry
    phantom : Phantom
        object used for generating the sinogram
    spec1 : Spectrum
        First polychromatic x-ray spectrum
    spec2 : Spectrum
        Second polychromatic x-ray spectrum
    D1 : float
        Dose assigned to spec1 acquisition [mGy].
    D2 : float
        Dose assigned to spec2 acquisition [mGy].
    N_matrix : int
        Number of pixels in the reconstructed matrix, shape [N_matrix, N_matrix]
    FOV : float
        Size of field-of-view to reconstruct, units cm.
    ramp : float, 0 to 1
        Cutoff fraction of Nyquist frequency for the recon filter.
    n_iters : int, optional
        Number of Gauss-Newton iterations for the mat decomp. The default is 30.
    mask_thresh : float, 0 to 1, optional
        Percent of maximum threshold for removing NaNs in sinos. The default is 0.95.

    Returns
    -------
    Sino_aa : 3D numpy array
        The two basis material sinograms.

    '''
    out_dir = get_out_dir_de(ct, phantom, spec1, spec2, D1, D2)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    # read the input single-energy sinograms
    sino_fname1 = get_out_dir_se(ct, phantom, spec1, D1)+'sino_raw.bin'
    sino_fname2 = get_out_dir_se(ct, phantom, spec2, D2)+'sino_raw.bin'
    sino1 = np.fromfile(sino_fname1, dtype=np.float32).reshape([ct.N_proj, ct.N_channels])
    sino2 = np.fromfile(sino_fname2, dtype=np.float32).reshape([ct.N_proj, ct.N_channels])
    
    # make sino mask, to threshold out the outliers after mat decomp below
    sino_mask = np.zeros(sino1.shape, dtype=bool)
    sino_mask[sino1>=mask_thresh*np.max(sino1)] = 1
 
    # call Gauss-Newton material decomposition algorithm
    Sino_aa = do_matdecomp_gn(sino1, sino2, spec1, spec2, n_iters)

    # iterate over the two basis materials
    for i in range(2):
        
        # threshold pixels outside the body, usually NaNs in the air
        Sino_aa[:,:,i][sino_mask] = 0
        sino_mat = Sino_aa[:,:,i]

        # save basis material sinogram
        sino_mat.astype(np.float32).tofile(out_dir + f'mat{i+1}_sino.bin')
        
        # reconstruct and save basis material image
        get_recon(out_dir, sino_mat, ct, N_matrix, FOV, ramp, use_gpu=GPU, HU=False, name=f'mat{i+1}_')
        
    return Sino_aa

