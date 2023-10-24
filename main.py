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

Change the global param `GPU` to False if there is no Nvidia GPU available.
"""


from system import Spectrum, ScannerGeometry, Phantom
from helpers import get_out_dir_se, get_sino, get_recon, get_matdecomp



if __name__ == '__main__':
    
    ####################################################################
    ### START OF INPUTS
    
    # recon parameters
    N_matrix = 512  # num pixels in recon matrix
    FOV = 50.0      # cm
    ramp = 0.8      # cutoff % of Nyquist frequency

    # geometry 
    ct = ScannerGeometry(N_channels=800, N_proj=1200, gamma_fan=0.8230337,   # 47 deg
                         SID=60.0, SDD=100.0,  pxshape='rect', eid=True, 
                         detector_file='input/detector/eta_eid_mv.bin')  
    
    # phantom 
    id_phantom = 'cylinder_5mats'
    filename_phantom = 'input/phantom/cylinder_5mats_512x512_uint8.bin'
    filename_matcomp = 'input/phantom/cylinder_5mats_matcomp.txt'
    phantom = Phantom(id_phantom, filename_phantom, filename_matcomp, Nx=512, Ny=512, Nz=1, ind=0) 
 
    # spectra -- can choose any dual energy combo from these options
    filepath_spectrum = 'input/spectrum/'
    filename_dict = {
        '6MV':       "Accuray_treatment6MV.csv", 
        'detunedMV': "Accuray_detuned.csv", 
        '80kV':      "spec80.mat", 
        '120kV':     "spec120.mat", 
        '140kV':     "spec140.mat"     }

    ### END OF INPUTS    
    ####################################################################

   
    import matplotlib.pyplot as plt

    # main loop over all the dual energy spectral combinations
    #     spec_id1, spec_id2 -- spectrum names (see filename_dict above)
    #     D1, D2 -- corresponding dose in mGy
    for spec_id1, spec_id2, D1, D2 in [['140kV', '80kV', 5, 5]]:

        ### 1st -- single energy acquisitions with each spectrum
        for spec_id, dose in [[spec_id1, D1], [spec_id2, D2]]:
     
            dose = 1e-3
            # load spectrum with dose initialized at 1 mGy before scaling
            spec = Spectrum(filepath_spectrum+filename_dict[spec_id], spec_id)
            spec.rescale_I0(ct.A_iso * dose / ct.N_proj)  # rescale by pixel area / number of views
                
            out_dir = get_out_dir_se(ct, phantom, spec, dose)
            print(f'\n{out_dir}')

            # forward project
            sino_raw, sino_log = get_sino(out_dir, ct, phantom, spec)    
            fig, ax = plt.subplots(1, 2, dpi=300, figsize=[9,5])
            m = ax[0].imshow(sino_raw, cmap='gray')
            fig.colorbar(m, ax=ax[0])
            m = ax[1].imshow(sino_log, cmap='gray')
            fig.colorbar(m, ax=ax[1])
            fig.tight_layout()
            plt.show()

            # back project
            recon_HU = get_recon(out_dir, sino_log, ct, N_matrix, FOV, ramp, HU=True, spec=spec)
            fig, ax = plt.subplots(1, 1, dpi=300, figsize=[6,5])
            plt.imshow(recon_HU, cmap='gray')
            plt.colorbar()
            plt.show()

        ### 2nd -- do material decomposition
        spec1 = Spectrum(filepath_spectrum+filename_dict[spec_id1], spec_id1)
        spec2 = Spectrum(filepath_spectrum+filename_dict[spec_id2], spec_id2)
        spec1.rescale_I0(ct.A_iso * D1 / ct.N_proj) 
        spec2.rescale_I0(ct.A_iso * D2 / ct.N_proj) 
            
        # do material decomposition into ICRU tissue and bone
        get_matdecomp(ct, phantom, spec1, spec2, D1, D2, N_matrix, FOV, ramp, n_iters=50)
    
        print(f'matdecomp finished for {spec_id1}-{spec_id2}') 
    





