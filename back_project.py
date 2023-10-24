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
    pre-process the projections for recon
    applies fan-beam filter and sinc window for noise suppression
    """
    
    gammas = ct.gammas - ct.dgamma_channel/2  # why need to subtract ???
    if ct.N_proj%2==1:   # this is buggy :-( 
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
    """Get the coordinates needed for the reconstruction matrix (common to all recons with same matrix)"""
    ## matrix coordinates: (r, theta)
    sz = FOV/N_matrix 
    matrix_coord_1d = cp.arange((1-N_matrix)*sz/2, N_matrix*sz/2, sz, dtype=cp.float32)
    X_matrix, Y_matrix = cp.meshgrid(matrix_coord_1d, -matrix_coord_1d)
    r_matrix = cp.sqrt(X_matrix**2 + Y_matrix**2)
    theta_matrix = cp.arctan2(X_matrix, Y_matrix) + cp.pi   
    return r_matrix, theta_matrix


def do_recon(sino, r_matrix, theta_matrix, SID, dgamma, dbeta):

    N_proj, N_cols = sino.shape
    N_matrix, _ = r_matrix.shape
    gamma_max = dgamma*N_cols/2

    matrix = cp.zeros([N_matrix, N_matrix], dtype=cp.float32)

    t0 = time()
    for i_proj in range(N_proj):  # create the fbp for each projection view i
        if i_proj%100 == 0:            
            print(f'{i_proj} / {N_proj}, t={time() - t0:.2f}s')

        beta = i_proj*dbeta          
        gamma_targets = cp.arctan(r_matrix*cp.cos(beta - theta_matrix) / (r_matrix*cp.sin(beta - theta_matrix) + SID))
        L2_M = r_matrix**2 * cp.cos(beta - theta_matrix)**2 + (SID + r_matrix*cp.sin(beta - theta_matrix))**2
        
        i_gamma0_matrix = ((gamma_targets + gamma_max)/dgamma).astype(cp.int32)   # matrix of indices (between 0 and N_cols-1) corresponding to sinogram pixels in row i_proj
        fbp_i = cp.choose(i_gamma0_matrix, sino[i_proj], mode='clip')  # might want to lerp i_proj and i_proj+1 !!!
        fbp_i[cp.abs(gamma_targets).get() > gamma_max] = 0
        
        matrix += fbp_i * dbeta / L2_M

    return matrix.get()




