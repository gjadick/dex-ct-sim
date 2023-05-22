#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy.linalg import inv
import xcompy as xc
import time


# basis material parameters
mat1 = 'ICRU tissue'
matcomp1 = 'H(10.2)C(14.3)N(3.4)O(70.8)Na(0.2)P(0.3)S(0.3)Cl(0.2)K(0.3)'  
density1 = 1.06 #g/cm3
mat2 = 'ICRU bone'
matcomp2 = 'H(3.4)C(15.5)N(4.2)O(43.5)Na(0.1)Mg(0.2)P(10.3)S(0.3)Ca(22.5)' 
density2 = 1.92 # g/cm3


def optimize_sino(Sino_gg, ee, i0, mus, n_iters, verbose=True):
    """
    This function applies Newton Iterations to the sinogram to solve the
    material estimation problem. It is the first update in the ADMM scheme,
    where the primal variable is updated.
    
    Parameters
    ----------
    Sino_gg : (nMeas, nViews, nBins) numpy ndarray 
        Sinogran in counts
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
            
            Sino_aa[j,:,:] -= (inv(H_poiss.transpose([2,0,1])) * (dF_poiss.T)[:,np.newaxis,:]).sum(axis=2)

    return Sino_aa



def detresponse(E, ideal=False, eid=True, detector_file='input/detector/eta.npy'):
    if ideal:
        eta_E = 1.0
    else:
        data = np.fromfile(detector_file, dtype=np.float32)
        N_det_energy = len(data)//2
        det_E = data[:N_det_energy]      # 1st half is energies
        det_eta_E = data[N_det_energy:]    # 2nd half is detective efficiencies
        eta_E = np.interp(E, det_E, det_eta_E)  # interp file to target energies
    
    if eid:
        return E*eta_E  
    else:
        return eta_E
    
    
    
def do_matdecomp_gn(sino1, sino2, spec1, spec2, id_phantom, n_iters):
    
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
    i01 = np.interp(ee, spec1.E, spec1.I0) * detresponse(ee) * dE #* ee
    i02 = np.interp(ee, spec2.E, spec2.I0) * detresponse(ee) * dE #* ee
    i0 = np.tile([i01, i02], (1, N_channels)).reshape([2, N_channels, N_energies])
    
    # mus: (nMats, nEnergies) numpy ndarray
    # linear attenuation table in [1/cm]
    mus = []
    for density, matcomp in [[density1, matcomp1], [density2, matcomp2]]:
        #u_E = density * xc.mixatten(matcomp, ee)
        u_E = xc.mixatten(matcomp, ee)
        mus.append(u_E)
    mus = np.array(mus)
    
    Sino_aa =  optimize_sino(Sino_gg, ee, i0, mus, n_iters)
    return Sino_aa
    

        





