#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cupy as cp
import xcompy as xc
import time


# Hardcoded basis materials.
mat1 = 'ICRU tissue'
matcomp1 = 'H(10.2)C(14.3)N(3.4)O(70.8)Na(0.2)P(0.3)S(0.3)Cl(0.2)K(0.3)'  
density1 = 1.06  # [g/cm3]
mat2 = 'ICRU bone'
matcomp2 = 'H(3.4)C(15.5)N(4.2)O(43.5)Na(0.1)Mg(0.2)P(10.3)S(0.3)Ca(22.5)' 
density2 = 1.92  # [g/cm3]


def optimize_sino(Sino_gg, ee, i0, mus, n_iters, verbose=True, dtype=cp.float32):
    """
    This function applies Newton Iterations to the sinogram to solve the
    material estimation problem. It is the first update in the ADMM scheme,
    where the primal variable is updated.
    
    This uses cupy instead of numpy, but seems to give different results :-(
    Parameters
    ----------
    Sino_gg : (nMeas, nViews, nBins) numpy ndarray 
        Sinogram in counts
    ee : (nEnergies) numpy ndarray 
        Energies of the specturm [keV]
    i0 : (nMeas, nBins, nEnergies) numpy ndarray
        Spectrum in counts with shape [nMeas,nBins,nEnergies]
    mus : (nMats, nEnergies) numpy ndarray
        Linear attenuation table in [1/cm]
    nIters : int
        The number if iterations in the optimiaztion algorithm

    Returns
    -------
    Sino_aa : (nViews, nBins, nMats) numpy ndarray 
    """
    
    Sino_gg = cp.array(Sino_gg, dtype=dtype)
    ee = cp.array(ee, dtype=dtype)
    i0 = cp.array(i0, dtype=dtype)
    mus = cp.array(mus, dtype=dtype)

    n_meas, nViews, nBins = Sino_gg.shape
    nMats = mus.shape[0]

    #[nViews, nBins, nMats]    
    EPS = 1e-6
    Sino_aa = cp.full([nViews,nBins,nMats],EPS).astype(dtype)   ## aa = density line integrals
    
    #[nMeas, nMats, nBins, nEnergies]        
    ssff = i0[:,cp.newaxis,:,:]*mus[cp.newaxis,:,cp.newaxis,:]

    #[nMeas, nMats, nMats, nBins, nEnergies]         
    ssff2 = i0[:,cp.newaxis,cp.newaxis,:,:] * (mus[cp.newaxis,:,:] * mus[:,cp.newaxis,:])[cp.newaxis,:,:,cp.newaxis,:]
        
    t0 = time.time()

    for j in range(nViews):
      
        if j%20==0 and verbose:
            print(j, '/', nViews, f't={time.time() - t0:.2f}s')
        
        for k in range(n_iters):

            atten = cp.exp(cp.sum(-1.0*Sino_aa[j,:,:][:,:,cp.newaxis]*mus, axis=1).clip(-700,700))

            nu = cp.sum(i0*atten,axis=2)
            nu_grad = -1.0*cp.sum(ssff*atten, axis=3)
            nu_hess = cp.sum(ssff2*atten, axis=4)

            dF_poiss = -1.0*cp.sum((Sino_gg[:,j,:]/nu-1.0)[:,cp.newaxis,:]*nu_grad, axis=0)
            H_poiss = -1.0*cp.sum((Sino_gg[:,j,:]/nu-1.0)[:,cp.newaxis,cp.newaxis,:]*nu_hess 
                                - (Sino_gg[:,j,:]/(nu*nu))[:,cp.newaxis,cp.newaxis,:]*(nu_grad[:,cp.newaxis,:,:]*nu_grad[:,:,cp.newaxis,:]), axis=0)
            
            Sino_aa[j,:,:] -= (cp.linalg.inv(H_poiss.transpose([2,0,1])) * (dF_poiss.T)[:,cp.newaxis,:]).sum(axis=2)

    return Sino_aa.get()


def optimize_sino_cpu(Sino_gg, ee, i0, mus, n_iters, verbose=True):
    """
    Copy of optimize_sino() but using numpy arrays (cpu) instead of cupy (gpu).
    Cupy gives nonsense results, need to investigate why there is a difference.
    Using cpu instead of gpu increases runtime ~ 8x but results in proper output. 
    See optimize_sino.__doc__ for full documentation.
    """
    n_meas, nViews, nBins = Sino_gg.shape
    nMats = mus.shape[0]

    #[nViews, nBins, nMats]    
    EPS = 1e-6
    Sino_aa = np.full([nViews,nBins,nMats],EPS)   ## aa = density line integrals

    #[nMeas, nMats, nBins, nEnergies]        
    ssff = i0[:,np.newaxis,:,:]*mus[np.newaxis,:,np.newaxis,:]

    #[nMeas, nMats, nMats, nBins, nEnergies]         
    ssff2 = i0[:,np.newaxis,np.newaxis,:,:] * (mus[np.newaxis,:,:] * mus[:,np.newaxis,:])[np.newaxis,:,:,np.newaxis,:]

    t0 = time.time()

    for j in range(nViews):

        if j%20==0 and verbose:
            print(j, '/', nViews, f't={time.time() - t0:.2f}s')

        for k in range(n_iters):

            atten = np.exp(np.sum(-1.0*Sino_aa[j,:,:][:,:,np.newaxis]*mus, axis=1).clip(-700,700))

            nu = np.sum(i0*atten,axis=2)
            nu_grad = -1.0*np.sum(ssff*atten, axis=3)
            nu_hess = np.sum(ssff2*atten, axis=4)

            dF_poiss = -1.0*np.sum((Sino_gg[:,j,:]/nu-1.0)[:,np.newaxis,:]*nu_grad, axis=0)
            H_poiss = -1.0*np.sum((Sino_gg[:,j,:]/nu-1.0)[:,np.newaxis,np.newaxis,:]*nu_hess - (Sino_gg[:,j,:]/(nu*nu))[:,np.newaxis,np.newaxis,:]*(nu_grad[:,np.newaxis,:,:]*nu_grad[:,:,np.newaxis,:]), axis=0)

            Sino_aa[j,:,:] -= (np.linalg.inv(H_poiss.transpose([2,0,1])) * (dF_poiss.T)[:,np.newaxis,:]).sum(axis=2)

    return Sino_aa

    
def do_matdecomp_gn(ct, sino1, sino2, spec1, spec2, n_iters):
    
    N_proj, N_channels = sino1.shape

    # Sino_gg: (nMeas, nViews, nBins) numpy ndarray
    # sinogram in counts
    Sino_gg = np.array([sino1, sino2])
    
    # ee: (nEnergies) numpy ndarray
    # energies of the spectrum [keV]  ~ intervals of 1 keV required?
    ee = np.array(sorted(list(set(np.append(spec1.E, spec2.E)))))        
    N_energies = len(ee)
    dE = np.append([ee[0]], ee[1:]-ee[:-1]) # 1st energy bin is 0 to E[0]

    # i0: (nMeas, nBins, nEnergies) numpy ndarray
    # spectrum in counts *** multiplied by detector response 
    detresponse = np.array(np.interp(ee, ct.det_E, ct.det_eta_E))  
    if ct.eid:
        detresponse = detresponse * ee   
    i01 = np.interp(ee, spec1.E, spec1.I0) * detresponse * dE 
    i02 = np.interp(ee, spec2.E, spec2.I0) * detresponse * dE 
    i0 = np.tile([i01, i02], (1, N_channels)).reshape([2, N_channels, N_energies])
    
    # mus: (nMats, nEnergies) numpy ndarray
    # linear attenuation table in [1/cm]
    mus = []
    for density, matcomp in [[density1, matcomp1], [density2, matcomp2]]:
        #u_E = density * xc.mixatten(matcomp, ee)
        u_E = xc.mixatten(matcomp, ee)
        mus.append(u_E)
    mus = np.array(mus)
    
    #Sino_aa = optimize_sino(Sino_gg, ee, i0, mus, n_iters)
    Sino_aa = optimize_sino_cpu(Sino_gg, ee, i0, mus, n_iters)
    return Sino_aa
    

def get_basismat_sinos(ct, sino_raw_1, sino_raw_2, spec1, spec2, n_iters=30, mask_thresh=0.95):
    '''
    For a DE-CT acquisition, make basis material sinograms using a
    Gauss-Newton algorithm and reconstruct into basis material images.The 
    basis materials are ICRU tissue (mat1) and bone (mat2). If desired, 
    these can be changed in matdecomp.do_matdecomp_gn().
    
    Parameters
    ----------
    ct : ScannerGeometry
        collection of parameters defining the CT acquisition geometry
    sino_raw_1,2 : Spectrum
        The two raw sinograms in units of photon counts.
    spec1,2 : Spectrum
        The two polychromatic x-ray spectra with counts scaled to the original dose.
    n_iters : int, optional
        Number of Gauss-Newton iterations for the mat decomp. The default is 30.
    mask_thresh : float, 0 to 1, optional
        Percent of maximum threshold for removing NaNs in sinos. The default is 0.95.

    Returns
    -------
    sino_mat1, sino_mat2 : 2D numpy arrays
        The two basis material sinograms.

    '''

    # make sino mask, to threshold out the outliers after mat decomp below
    sino_mask = np.zeros(sino_raw_1.shape, dtype=bool)
    sino_mask[sino_raw_1>=mask_thresh*np.max(sino_raw_1)] = 1
 
    # call Gauss-Newton material decomposition algorithm
    density_line_integrals = do_matdecomp_gn(ct, sino_raw_1, sino_raw_2, spec1, spec2, n_iters)
    sino_mat1 = density_line_integrals[:,:,0]
    sino_mat2 = density_line_integrals[:,:,1]
    
    # apply the mask
    sino_mat1[sino_mask] = 0
    sino_mat2[sino_mask] = 0

    return sino_mat1, sino_mat2






