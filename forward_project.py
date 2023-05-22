#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:30:56 2022

@author: gjadick
"""

import numpy as np
import time
import xcompy as xc
import os


def siddons(src, trg, N, dR, EPS=1e-8):
    '''
    An implementation of Siddons raytracing.
    Assume 2D matrix, src & trg are outside of matrix.
    The line from src -> trg should intersect the matrix.
    
    Inputs:
        src : 
            [x1,y1], coordinates of the ray start point.
        trg :
            [x2,y2], coordinates of the ray end point.
        N :
            number of pixels in the matrix. (assume square)
        dR :
            size of pixels in the matrix. (assume isotropic)
    '''
    
    # take x, y of src and trg
    x1, y1 = src[0], src[1]
    x2, y2 = trg[0], trg[1]
    
    # number of planes in x and y directions
    Nx = N+1
    Ny = N+1
    
    # distance traversed over the x and y directions
    dx = x2 - x1 
    dy = y2 - y1

    #check for 0 distances
    if np.abs(dx) < EPS:
        dx = EPS
    if np.abs(dy) < EPS:
        dy = EPS

    coord_plane_1 = -N*dR//2

    def coord_plane(i):              # for i from 1 to N
        return coord_plane_1 + (i-1)*dR

    def aX(i):
        return (coord_plane(i) - x1) / dx

    def aY(i):
        return (coord_plane(i) - y1) / dy

    # parametric coords for intersections of ray with matrix
    a_min = max(min(aX(1), aX(Nx)), min(aY(1), aY(Ny)))
    a_max = min(max(aX(1), aX(Nx)), max(aY(1), aY(Ny)))

    # start and end index for i,j
    if dx >= 0:  # moving right
        i_min = Nx - (coord_plane(Nx) - a_min*dx - x1)/dR
        i_max = 1 + (x1 + a_max*dx - coord_plane_1)/dR
    else:
        i_min = Nx - (coord_plane(Nx) - a_max*dx - x1)/dR
        i_max = 1 + (x1 + a_min*dx - coord_plane_1)/dR

    if dy >= 0:  # moving up
        j_min = Ny - (coord_plane(Ny) - a_min*dy - y1)/dR
        j_max = 1 + (y1 + a_max*dy - coord_plane_1)/dR
    else:
        j_min = Ny - (coord_plane(Ny) - a_max*dy - y1)/dR
        j_max = 1 + (y1 + a_min*dy - coord_plane_1)/dR

    # round up/down for min/max, if not integer.
    if np.abs(j_min - np.round(j_min)) > EPS:
        j_min = np.ceil(j_min)
    if np.abs(j_max - np.round(j_max)) > EPS:
        j_max = np.floor(j_max)

    if np.abs(i_min - np.round(i_min)) > EPS:
        i_min = np.ceil(i_min)
    if np.abs(i_max - np.round(i_max)) > EPS:
        i_max = np.floor(i_max)

    # cast as integers
    i_min = np.round(i_min).astype(int)
    i_max = np.round(i_max).astype(int)
    j_min = np.round(j_min).astype(int)
    j_max = np.round(j_max).astype(int)

    i_vals = list(range(i_min, i_max+1, 1))
    j_vals = list(range(j_min, j_max+1, 1))

    # arrange a_x, a_y in ascending order
    if dx >= 0: 
        a_x = [aX(i) for i in i_vals]
    else:
        a_x = [aX(i) for i in i_vals[::-1]]

    if dy >= 0: 
        a_y = [aY(j) for j in j_vals]
    else:
        a_y = [aY(j) for j in j_vals[::-1]]

    # merge a_x, a_y into sorted alphas
    i = 0
    j = 0
    alphas = np.zeros(len(a_x) + len(a_y))
    while i<len(a_x) or j<len(a_y):
        if i<len(a_x) and j<len(a_y):
            if a_x[i] < a_y[j]:
                alphas[i+j] = a_x[i]
                i+=1
            else:
                alphas[i+j] = a_y[j]
                j+=1
        elif i<len(a_x):
            alphas[i+j] = a_x[i]
            i+=1
        elif j<len(a_y):
            alphas[i+j] = a_y[j]
            j+=1

    # get the difference between alphas (normalized lengths)
    dalphas = alphas[1:] - alphas[:-1]
    dST = src - trg
    d12 = np.linalg.norm(dST, axis=-1)
    l = dalphas * d12

    # get the voxel indices [i, j]
    Nl = len(dalphas)
    il = np.zeros(Nl, dtype=int)
    jl = np.zeros(Nl, dtype=int)
    m = 0
    for m in range(0, Nl):
        il[m] = (x1 + 0.5*(alphas[m]+alphas[m+1])*dx - coord_plane_1)/dR
        jl[m] = (y1 + 0.5*(alphas[m]+alphas[m+1])*dy - coord_plane_1)/dR

        
    # do some post processing.
    # 1. In case there are any l=0, remove those values 
    # 2. Get rid of indices larger than N
    l, il, jl = l[l>EPS], il[l>EPS], jl[l>EPS]
    l, il, jl = l[il<N], il[il<N], jl[il<N]
    l, il, jl = l[jl<N], il[jl<N], jl[jl<N]
    
    return np.array([l, il, jl])



def detect_transmitted_sino(E, I0_E, sino_T_E, detector_file='input/detector/eta.npy',
                            ideal=False, noise=True, eid=True):
    '''
    Function to calculate noisy detected signal in a single sinogram pixel.

    Parameters
    ----------
    E : 1D np.array
        List of energy values in the spectrum (keV).
    I0_E : 1D np.array
        List of number of photons in each corresponding energy bin.
    T_E : 1D np.array
        % of photons transmitted after the phantom, exp(-u*L), in each energy bin.
    detector_file : str
        path to numpy float32 file with the detective efficiency data [ E, eta(E) ]
    ideal : bool, optional
        Whether the detection function is ideal (eta=1) or not. The default is False.
    noise : bool, optional
        Whether to add Poisson noise. The default is True.
    eid : bool, optional
        Whether to energy-weight the sum over all detected photons. The default is True.

    Returns
    -------
    signal : float
        Detected pixel value.

    '''
    # transmitted spectrum: I0_E * T_E in each pixel
    sino_transmitted = I0_E * sino_T_E   # shape: [N_proj, N_channels, N_energy]
    N_proj = sino_T_E.shape[0]
    N_cols = sino_T_E.shape[1]

    # get efficiency
    if ideal:
        eta_E = 1.0
    else:
        data = np.fromfile(detector_file, dtype=np.float32)
        N_det_energy = len(data)//2
        det_E = data[:N_det_energy]      # 1st half is energies
        det_eta_E = data[N_det_energy:]    # 2nd half is detective efficiencies
        eta_E = np.interp(E, det_E, det_eta_E)  # interp file to target energies
    
    # get energy bin sizes
    dE = np.append([E[0]], E[1:]-E[:-1]) # 1st energy bin is 0 to E[0]

    sino = np.zeros([N_proj, N_cols], dtype=np.float32)
    # add noise to each pixel
    for i_proj in range(N_proj):
        for i_gamma in range(N_cols):
            
            # number of photons counted in each energy bin
            N_photons = sino_transmitted[i_proj, i_gamma] * eta_E * dE 
            if noise: 
                N_photons = np.random.poisson(N_photons)   
            
            # if EID, add energy-weighting to the Poisson RVs
            if eid:
                signal_E = E*N_photons
            else:
                signal_E = N_photons
        
            # sum over all energy bins
            signal = np.sum(signal_E)    # bin width included! 
            
            sino[i_proj, i_gamma] = signal
            
    return sino


def get_atten(u_dict, attens, lengths):
    '''
    For a single pixel, converts attens/lengths to attenuation 
    in each energy bin

    Parameters
    ----------
    u_dict : dictionary
        The dictionary of mu(E) arrays corresponding to each integer material ID.
    attens : 1D array [int]
        The array of integer material IDs of interest.
    lengths : 1D array [float]
        The array of corresponding path lengths for each material ID.

    Returns
    -------
    uL_E : 1D array [float]
        The attenuation through the pixels as a funtion of energy.
    '''
    u_E = np.array([u_dict[mat_id] for mat_id in attens])
    uL_E = u_E.T @ lengths
    return np.exp(-uL_E)


def raytrace_fanbeam(ct, phantom, spec, u_dict, use_cache=True, verbose=True):

    # check whether a cached sino exists from previous run
    # the transmission through the phantom in spec's energy bins for given ct geometry 
    cache_dir  = f'./output/cache/{ct.geo_id}/{phantom.name}/{spec.name}/'
    cache_file = cache_dir + 'sino.npy'
    if use_cache and os.path.exists(cache_file):
        print('using cached transmission sino,', cache_dir)
        sino_T_E = np.fromfile(cache_file, dtype=np.float32).reshape([ ct.N_proj, ct.N_channels, len(spec.E)])
        
    # if not, raytrace
    else:
        print('raytracing, no cache file,', cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        sino_T_E = np.zeros([ ct.N_proj, ct.N_channels, len(spec.E)], dtype=np.float32)
        
        t0 = time.time()
        
        # rotate the source to each projection
        for i_proj, theta0 in enumerate(ct.thetas+np.pi):
            
            if i_proj%20==0 and verbose:
                print(i_proj, '/', ct.N_proj, f't={time.time() - t0:.2f}s')
            
            src =  ct.SID*np.array([np.cos(theta0), np.sin(theta0)])            
            for i_gamma, gamma in enumerate(ct.gammas):
                theta = theta0+gamma+ 1.5*ct.dgamma_channel # offset and center to match FBP
                trg =   src - ct.SDD*np.array([np.cos(theta), np.sin(theta)])
                
                # raytrace src -> trg using siddons alg
                lengths, xInd, yInd = siddons(src, trg, phantom.Nx, phantom.sx)
                
                ji_coords = np.array([yInd.astype(np.uint16), xInd.astype(np.uint16)]).T
                ji = np.ravel_multi_index(ji_coords.reshape(ji_coords.size//2, 2).T, phantom.M.shape)
                atten_ids = phantom.M.take(ji)
                    
                sino_T_E[i_proj, i_gamma] = get_atten(u_dict, atten_ids, lengths)
        
        # cache the calculated transmission sinogram
        sino_T_E.tofile(cache_file) 
        
    # process the transmitted energy information into an actual signal
    sino = detect_transmitted_sino(spec.E, spec.I0, sino_T_E)
    
    # fix div by 0?
    EPS = 1 #1e-8 
    sino[sino<EPS] = EPS
        
    return sino





