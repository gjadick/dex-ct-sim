#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 13:30:24 2022

@author: gjadick

Script for:
    (1) generating single-energy CT sinograms using Siddons raytracing
    (2) reconstructing CT images using fan-beam filtered back-projection
    (3) generating dual-energy CT basis material sinograms using Gauss-Newton 
        decomposition and reconstructing basis material images using FFBP.
"""

import os
import numpy as np

import xcompy as xc

from system import Spectrum, ScannerGeometry, Phantom, get_matcomp_dict
from forward_project import raytrace_fanbeam, detect_transmitted_sino
from back_project import pre_process, get_recon_coords, do_recon, do_recon_gpu
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

    # get the linear attenuation coefficients for the spectrum energies
    matcomp_dict = get_matcomp_dict(phantom.matcomp_filename)
    u_dict = {}
    for mat in matcomp_dict:
        density, matcomp = matcomp_dict[mat] 
        u_E = density * xc.mixatten(matcomp, spec.E)
        u_E[np.isnan(u_E)] = 0.0
        u_dict[mat] = u_E

    # forward project
    sino_raw = raytrace_fanbeam(ct, phantom, spec, u_dict)
    
    # log the data
    sino0 = np.ones([ct.N_proj, ct.N_channels])*detect_transmitted_sino(spec.E, spec.I0, np.ones([1,1,1]), noise=False)
    sino_log = np.log(sino0/sino_raw)

    # save sinos
    sino_raw.astype(np.float32).tofile(out_dir+'sino_raw.npy')
    sino_log.astype(np.float32).tofile(out_dir+'sino_log.npy')

    return sino_raw, sino_log


def get_recon(out_dir, sino_log, ct, N_matrix, FOV, ramp,  use_gpu=True, HU=False, spec=None, name=''):
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
    use_gpu : bool, optional
        Whether to use parallelization for recon. Requires PyCuda. 
        The default is True.
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

    ji_coord, r_M, theta_M, gamma_target_M, L2_M = get_recon_coords(
        N_matrix, FOV, ct.N_proj, ct.dtheta_proj, ct.SID)

    if use_gpu:
        recon = do_recon_gpu(sino_filtered, gamma_target_M, L2_M, ct.gammas, ct.dtheta_proj)
    else:
        recon = do_recon(sino_filtered, ct.dtheta_proj, ct.gammas,      
                 gamma_target_M, L2_M, ji_coord, verbose=True)

    # convert to HU
    if HU:
        if spec is None:
            print('HU conversion failed, spec must be assigned to get mu values')
        else:
            recon = 1000*(recon - spec.u_water)/(spec.u_water - spec.u_air)
    
    # save recons
    recon_fname = out_dir+f'{name}recon_{N_matrix}_{int(FOV):02}cm_{int(ramp*100):03}ramp.npy' 
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
    sino_fname1 = get_out_dir_se(ct, phantom, spec1, D1)+'sino_raw.npy'
    sino_fname2 = get_out_dir_se(ct, phantom, spec2, D2)+'sino_raw.npy'
    sino1 = np.fromfile(sino_fname1, dtype=np.float32).reshape([ct.N_proj, ct.N_channels])
    sino2 = np.fromfile(sino_fname2, dtype=np.float32).reshape([ct.N_proj, ct.N_channels])
    
    # make sino mask, to threshold out the outliers after mat decomp below
    sino_mask = np.zeros(sino1.shape, dtype=bool)
    sino_mask[sino1>=mask_thresh*np.max(sino1)] = 1
 
    # call Gauss-Newton material decomposition algorithm
    Sino_aa = do_matdecomp_gn(sino1, sino2, spec1, spec2, id_phantom, n_iters)

    # iterate over the two basis materials
    for i in range(2):
        
        # threshold pixels outside the body, usually NaNs in the air
        Sino_aa[:,:,i][sino_mask] = 0
        sino_mat = Sino_aa[:,:,i]

        # save basis material sinogram
        sino_mat.astype(np.float32).tofile(out_dir + f'mat{i+1}_sino.npy')
        
        # reconstruct and save basis material image
        get_recon(out_dir, sino_mat, ct, N_matrix, FOV, ramp, HU=False, name=f'mat{i+1}_')
        
    return Sino_aa




if __name__ == '__main__':
        
    # recon parameters
    N_matrix = 512  # num pixels in recon matrix
    FOV = 50.0      # cm
    ramp = 0.8      # cutoff % of Nyquist frequency
    
    # geometry 
    ct = ScannerGeometry(N_channels=800, N_proj=1200, gamma_fan=0.8230337,   # 47 deg
                      SID=60.0, SDD=100.0, pxshape='rect')  
    
    # phantoms 
    phantoms = []
    for id_phantom in ['pelvis', 'pelvis_metal']:
        filename_phantom = f'input/phantom/xcat_{id_phantom}_uint8_512_512_1_1mm.bin'
        filename_matcomp = 'input/phantom/xcat_elemental_comps.txt'
        phantom = Phantom(filename_phantom, id_phantom, filename_matcomp, 512, 512, 1, ind=0) 
        phantoms.append(phantom)
 
    # spectra
    filepath_spectrum = 'input/spectrum/'
    filename_dict = {
        '6MV':       "Accuray_treatment6MV.csv", 
        'detunedMV': "Accuray_detuned.csv", 
        '80kV':      "spec80.mat", 
        '120kV':     "spec120.mat", 
        '140kV':     "spec140.mat"     }
    
        
    # 1st -- single-energy CT acquisitions
    for phantom in phantoms: 
        for spec_id, doses in [['80kV',   [1, 5, 10]],
                                ['120kV', [10]], 
                                ['140kV', [5, 10]], 
                                ['detunedMV', [9, 10]]
                               ]: 
            for dose in doses:  
                
                # reload spectrum, so dose is initialized at 1 mGy before scaling
                spec = Spectrum(filepath_spectrum+filename_dict[spec_id], spec_id)
                spec.rescale_I0(ct.A_iso * dose / ct.N_proj)  # rescale by pixel area / number of views
                
                out_dir = get_out_dir_se(ct, phantom, spec, dose)
                print('\n', out_dir)

                sino_raw, sino_log = get_sino(out_dir, ct, phantom, spec)    
                recon_HU = get_recon(out_dir, sino_log, ct, N_matrix, FOV, ramp, HU=True, spec=spec)

                
    # 2nd -- dual-energy CT material decomposition
    # (this requires existing SE-CT output files for each spectrum/dose pair)
    for phantom in phantoms:
        for spec_id1, spec_id2, D1, D2 in [
                                   #['140kV', '80kV', 5, 5],
                                   #['detunedMV', '80kV', 9, 1],
                                  ]:
            
            # reload spectra, so doses are initialized at 1 mGy before scaling
            spec1 = Spectrum(filepath_spectrum+filename_dict[spec_id1], spec_id1)
            spec2 = Spectrum(filepath_spectrum+filename_dict[spec_id2], spec_id2)
            spec1.rescale_I0(ct.A_iso * D1 / ct.N_proj) 
            spec2.rescale_I0(ct.A_iso * D2 / ct.N_proj) 
            
            # do material decomposition into ICRU tissue and bone
            get_matdecomp(ct, phantom, spec1, spec2, D1, D2, N_matrix, FOV, ramp, n_iters=50)
    
    
    
    
    
    
    
        