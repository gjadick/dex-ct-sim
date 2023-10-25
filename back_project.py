#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:03:45 2022

@author: gjadick
"""

import numpy as np
import cupy as cp
from time import time
    

def pre_process(sino_log, ct, ramp_percent):
    """
    Pre-process the raw projections for FFBP.
    Applies fan-beam filter and sinc window for noise suppression.

    Parameters
    ----------
    sino_log : 2D cupy array
        The log'd raw sinogram data, ln(-I/I0).
    ct : ScannerGeometry
        The geometry used for the raw data acquisition.
    ramp_percent : float
        Percent cutoff of the Nyquist frequency for sinc windowing.

    Returns
    -------
    sino_filtered : 2D cupy array
        The pre-processed sinogram (same shape as input sino_log).
    """

    gammas = ct.gammas - ct.dgamma_channel/2  
    if ct.N_proj%2==1:   # may cause issues with odd num projections !!!
        n = cp.arange(-ct.N_channels//2+1, ct.N_channels, 1, dtype=cp.float32)
    else:
        n = cp.arange(-ct.N_channels//2, ct.N_channels, 1, dtype=cp.float32)
        
    # modified fanbeam data (q --> qm)
    sino_qm = np.array([q*ct.SID*np.cos(gammas) for q in sino_log])    # with cosine weighting
    sino_qm = cp.array(sino_qm, dtype=cp.float32)
    
    # # Hsieh method for fanbeam ramp filter
    g = cp.zeros(ct.N_channels, dtype=cp.float32)
    for i in range(ct.N_channels):
        if n[i]==0:
            g[i] = 1/(8 * ct.dgamma_channel**2)
        elif n[i]%2==1: # odd
            g[i] = -0.5/(np.pi * np.sin(gammas[i]))**2

    # implement sinc window (Hsieh CT 4th edition, eq. 3.31 )
    G = cp.fft.fft(g)
    w = cp.fft.fftfreq(g.size)  # frequencies
    window = cp.zeros(ct.N_channels, dtype=cp.float32)
    wL, wH = 0, ramp_percent*cp.max(w)
    for i in range(ct.N_channels):
        if np.abs(w[i]) <= wL:
            window[i] = 1.0
        elif wL < cp.abs(w[i]) and cp.abs(w[i]) <= wH:
            frac = np.pi*(cp.abs(w[i])-wL)/(wH-wL)
            window[i] = cp.sin(frac)/frac
            
    g = cp.real(cp.fft.ifft(G*window))

    # convolve and scale by dgamma
    sino_filtered = cp.array([cp.convolve(qm, g, mode='same') for qm in sino_qm], dtype=cp.float32)        
    sino_filtered *= ct.dgamma_channel
    
    return sino_filtered
        

def get_recon_coords(N_matrix, FOV):
    """
    Compute the polar coordinates corresponding to each pixel in the final 
    reconstruction matrix (common to all recons with same matrix dimensions.)

    Parameters
    ----------
    N_matrix : int
        Number of pixels in the recon matrix so that its shape = [N_matrix, N_matrix].
    FOV : float
        Field of view of the recon matrix [cm].

    Returns
    -------
    r_matrix : 2D cupy array ~ [N_matrix, N_matrix]
        Radial coordinate [cm] of each pixel.
    theta_matrix : 2D cupy array ~ [N_matrix, N_matrix]
        Angle coordinate [radians] of each pixel.

    """
    sz = FOV/N_matrix  # pixel size
    matrix_coord_1d = cp.arange((1-N_matrix)*sz/2, N_matrix*sz/2, sz, dtype=cp.float32)
    X_matrix, Y_matrix = cp.meshgrid(matrix_coord_1d, -matrix_coord_1d)
    r_matrix = cp.sqrt(X_matrix**2 + Y_matrix**2)
    theta_matrix = cp.arctan2(X_matrix, Y_matrix) + cp.pi   
    return r_matrix, theta_matrix


def do_recon(sino, r_matrix, theta_matrix, SID, dgamma, dbeta):
    """
    Reconstruct a log'd sinogram using fan-beam filtered back-projection.

    Parameters
    ----------
    sino_log : 2D cupy array ~ [N_proj, N_col]
        The log'd sinogram data, ln(-I/I0). Should already be pre-processed.
        Shape is the number of projection views (N_proj) by number of detector
        channels (N_col).
    r_matrix : 2D cupy array ~ [N_matrix, N_matrix]
        Radial coordinate [cm] of each pixel.
    theta_matrix : 2D cupy array ~ [N_matrix, N_matrix]
        Angle coordinate [radians] of each pixel.
    SID : float
        Source-to-isocenter distance [cm].
    dgamma : float
        Angular decrement [radians] between adjacent detector channels.
    dbeta : TYPE
        Angular decrement [radians] between projection views.

    Returns
    -------
    matrix : 2D numpy array ~ [N_matrix, N_matrix]
        The reconstructed CT image.
    """

    N_proj, N_cols = sino.shape
    N_matrix, _ = r_matrix.shape
    gamma_max = dgamma*N_cols/2

    matrix = cp.zeros([N_matrix, N_matrix], dtype=cp.float32)

    t0 = time()
    for i_proj in range(N_proj):  # create the fbp for each projection view i
        if i_proj%100 == 0:            
            print(f'{i_proj} / {N_proj}, t={time() - t0:.2f}s')

        beta = i_proj*dbeta  # angle to x-ray source for each projection      
        gamma_targets = cp.arctan(r_matrix*cp.cos(beta - theta_matrix) / (r_matrix*cp.sin(beta - theta_matrix) + SID))
        L2_M = r_matrix**2 * cp.cos(beta - theta_matrix)**2 + (SID + r_matrix*cp.sin(beta - theta_matrix))**2
        
        i_gamma0_matrix = ((gamma_targets + gamma_max)/dgamma).astype(cp.int32)   # matrix of indices (between 0 and N_cols-1) corresponding to sinogram pixels in row i_proj
        fbp_i = cp.choose(i_gamma0_matrix, sino[i_proj], mode='clip')  # might want to lerp i_proj and i_proj+1 !!!
        fbp_i[cp.abs(gamma_targets).get() > gamma_max] = 0
        
        matrix += fbp_i * dbeta / L2_M

    return matrix.get()




