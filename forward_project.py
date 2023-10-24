#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:30:56 2022

@author: gjadick
"""

import numpy as np
import cupy as cp
from time import time


siddons_2D_raw = cp.RawKernel( 
    name='siddons_2D_raw',
    code='''
    extern "C" __global__
    
    void siddons_2D_raw(int N_rays, float *src_x, float *src_y, float *trg_x, float *trg_y, float *mu, int N, float dr, float *a_x, float *a_y, float *alphas,  float *line_integral) 
    {
     
     int i = threadIdx.x + blockDim.x * blockIdx.x; //+ threadIdx.y + blockDim.y * blockIdx.y;
     
     if (i < N_rays) {
        float EPS = 1.0E-4;   // set relatively large for rounding later
        float x1 = src_x[i];  // source coordinates
        float y1 = src_y[i];
        float x2 = trg_x[i];  // target coordinates
        float y2 = trg_y[i];
        int Nx_pixels = N;    // assume square matrix
        int Ny_pixels = N; 
        int Nx = Nx_pixels + 1;  // number of planes in x and y directions
        int Ny = Ny_pixels + 1;
        int i0 = i*(N + 1);     // start index for pre-allocated a_x, a_y, alphas arrays
        float dx = x2 - x1;     // total distance traversed in x and y
        float dy = y2 - y1;
        float d12 = hypotf(dx, dy); 
        
        // Check to avoid div by zero error (lines parallel with either x, y axis).
        if ( fabsf(dx) < EPS) {dx = EPS;}
        if ( fabsf(dy) < EPS) {dy = EPS;}
        
        // Other key values (see Siddon's paper). 
        // The origin (0, 0) is defined in the center of the matrix.
        // The planes at pixel edges are indexed from 1 to Nx = N_pixels + 1.
        float coord_plane_1 = -N*dr / 2;  // assuming square for now
        float coord_plane_Nx = coord_plane_1 + Nx_pixels*dr;  
        float coord_plane_Ny = coord_plane_1 + Ny_pixels*dr;
        float aX_1 = (coord_plane_1 - x1) / dx;    // a ~ alpha
        float aX_Nx = (coord_plane_Nx - x1) / dx;
        float aY_1 = (coord_plane_1 - y1) / dy;
        float aY_Ny = (coord_plane_Ny - y1) / dy;
        
        // Parametric coords for intersections of ray with matrix.
        // Sometimes weird results for small N?
        float a_minf = fmaxf( fminf(aX_1, aX_Nx), fminf(aY_1, aY_Ny) );
        float a_maxf = fminf( fmaxf(aX_1, aX_Nx), fmaxf(aY_1, aY_Ny) );
        
        // Compute phantom intersection start and end indices in x,y directions.
        float i_minf, i_maxf, j_minf, j_maxf;
        if (dx >= 0.0) {   // moving right
            i_minf = Nx - (coord_plane_Nx - a_minf*dx - x1) / dr;
            i_maxf = 1 + (x1 + a_maxf*dx - coord_plane_1) / dr;
        }
        else {
            i_minf = Nx - (coord_plane_Nx - a_maxf*dx - x1) / dr;
            i_maxf = 1 + (x1 + a_minf*dx - coord_plane_1) / dr;
        }          
        if (dy >= 0.0) {   // moving up
            j_minf = Ny - (coord_plane_Ny - a_minf*dy - y1) / dr;
            j_maxf = 1 + (y1 + a_maxf*dy - coord_plane_1) / dr;
        }
        else {
            j_minf = Ny - (coord_plane_Ny - a_maxf*dy - y1) / dr;
            j_maxf = 1 + (y1 + a_minf*dy - coord_plane_1) / dr;
        }
        
        // Round up/down for min/max or if not round, cast as int.
        // If EPS is too small, this will cause some error.
        if ( fabsf(j_minf - nearbyintf(j_minf)) > EPS ) { j_minf = ceilf(j_minf); }
        if ( fabsf(j_maxf - nearbyintf(j_maxf)) > EPS ) { j_maxf = floorf(j_maxf); }
        if ( fabsf(i_minf - nearbyintf(i_minf)) > EPS ) { i_minf = ceilf(i_minf); }
        if ( fabsf(i_maxf - nearbyintf(i_maxf)) > EPS ) { i_maxf = floorf(i_maxf); }
        int i_min = lroundf(i_minf);
        int i_max = lroundf(i_maxf);
        int j_min = lroundf(j_minf);
        int j_max = lroundf(j_maxf);
        
        // Get parametric distances a_x, a_y in ascending order.
        int Na_x = i_max - i_min + 1;
        int Na_y = j_max - j_min + 1; 
        int Na = Na_x + Na_y;
      
        int ii, jj;  // i is reserved!
        float coord_plane_i;  
        for (ii = 0; ii < Na_x; ii++) {
            coord_plane_i = coord_plane_1 + (i_min + ii - 1)*dr; 
            if (dx >= 0) {  
                a_x[i0 + ii] = (coord_plane_i - x1) / dx; }
            else { 
                a_x[i0 + i_max - i_min - ii] = (coord_plane_i - x1) / dx; }  // go in reverse
        }
        for (jj = 0; jj < Na_y; jj++) {
            coord_plane_i = coord_plane_1 + (j_min + jj - 1)*dr;
            if (dy >= 0) {
                a_y[i0 + jj] = (coord_plane_i - y1) / dy; }
            else { 
                a_y[i0 + j_max - j_min - jj] = (coord_plane_i - y1) / dy; }  // go in reverse
        }  
        
        // Merge a_x and a_y into one parametric alphas array
        ii = 0;  // reset inds
        jj = 0;
        while (ii < Na_x || jj < Na_y) {
            if ( (ii < Na_x) && (jj < Na_y) ) {
                if (a_x[i0 + ii] < a_y[i0 + jj]) { 
                       alphas[2*i0 + ii + jj] = a_x[i0 + ii]; ii++; }                 
                else { alphas[2*i0 + ii + jj] = a_y[i0 + jj]; jj++; }  
            }
            else if (ii < Na_x) { alphas[2*i0 + ii + jj] = a_x[i0 + ii]; ii++; }  
            else if (jj < Na_y) { alphas[2*i0 + ii + jj] = a_y[i0 + jj]; jj++; }  
        }
    
        // Compute the final line integral.
        float max_length = sqrtf(2)*dr + EPS;
        float length;
        int il, jl;
        for (ii = 0; ii < Na-1; ii++) {  
            length = d12 * (alphas[2*i0 + ii + 1] - alphas[2*i0 + ii]);
            il = (x1 + 0.5*(alphas[2*i0 + ii + 1] + alphas[2*i0 + ii])*dx - coord_plane_1) / dr;
            jl = (y1 + 0.5*(alphas[2*i0 + ii + 1] + alphas[2*i0 + ii])*dy - coord_plane_1) / dr;
            if ((length>EPS && length<=max_length) && (il>=0 && il<N) && (jl>=0 && jl<N)) {  // check path is physical
                line_integral[i] = line_integral[i] + mu[il + N*jl]*length; 
            }
        }
    }}
                             
    '''
)

                             
def siddons_2D(src_x, src_y, trg_x, trg_y, matrix, sz_matrix_pixel, N_threads_max=1024):

    N_rays = src_x.size
    N_matrix = matrix.shape[0]
    # bits = (N_rays*5 + N_matrix**2 + 4*N_rays*(N_matrix+1)) * np.dtype(np.float32).itemsize
    # print(f'{N_rays} rays, {N_matrix} matrix = {bits/1e9:.4f} GB')
        
    # Initialize empty arrays to store parametric intersections for each ray
    a_x = cp.zeros(N_rays * (N_matrix+1), dtype=cp.float32)
    a_y = cp.zeros(N_rays * (N_matrix+1), dtype=cp.float32)
    alphas = cp.zeros(2*N_rays * (N_matrix+1), dtype=cp.float32)
    line_integrals = cp.zeros(N_rays, dtype=cp.float32)
    
    # Assign block/grid sizes (1D) and run.    
    siddons_2D_raw(block=(min(N_rays, N_threads_max), 1, 1), 
                   grid=(1 + N_rays//N_threads_max, 1, 1),
                   args=(N_rays, src_x, src_y, trg_x, trg_y, 
                         matrix, N_matrix, cp.float32(sz_matrix_pixel),
                         a_x, a_y, alphas, line_integrals))
    
    return line_integrals  # still on the device! 


def detect_transmitted_sino(E, I0_E, sino_T_E, ct, noise=True):
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
    ct : ScannerGeometry
        class with needed information about the detector and its efficiency
    noise : bool, optional
        Whether to add Poisson noise. The default is True.

    Returns
    -------
    signal : float
        Detected pixel value.

    '''
    eta_E = cp.array(np.interp(E, ct.det_E, ct.det_eta_E), dtype=cp.float32)  # detector efficiency at target energies
    dE = np.append([E[0]], E[1:]-E[:-1])  # energy bin widths, 1st is 0 to E[0]
    signal_E = cp.array(I0_E * dE) * eta_E * sino_T_E  # signal in each energy bin
    if noise:
        signal_E = cp.random.poisson(signal_E, dtype=cp.int32)  # might need to increase to int64?
    if ct.eid:
        signal_E = cp.array(E) * signal_E            
    return cp.sum(signal_E, axis=2) 


def raytrace_fanbeam(ct, phantom, spec):
    t0 = time()

    # Get coordinates for each source --> detector channel
    d_thetas = cp.tile(cp.array(ct.thetas + cp.pi, dtype=cp.float32)[:, cp.newaxis], ct.N_channels).ravel()  # use newaxis for correct tiling
    d_gammas = cp.tile(cp.array(ct.gammas, dtype=cp.float32), ct.N_proj)
    src_x = ct.SID * cp.cos(d_thetas)
    src_y = ct.SID * cp.sin(d_thetas)
    trg_x = src_x - ct.SDD * cp.cos(d_thetas + d_gammas)
    trg_y = src_y - ct.SDD * cp.sin(d_thetas + d_gammas)

    # For each monoenergy, raytrace for all the source, targets
    matrix_stack = phantom.M_mono_stack(spec.E) 
    sino_T_E = cp.zeros([ct.N_proj, ct.N_channels, len(spec.E)], dtype=np.float32)
    for i_energy, energy in enumerate(spec.E): 
        if i_energy%10==0:
            print(f'{i_energy} / {len(spec.E)}, t={time() - t0:.2f}s')
        uL_E = siddons_2D(src_x, src_y, trg_x, trg_y, matrix_stack[i_energy], phantom.sx)
        uL_E = uL_E.reshape([ct.N_proj, ct.N_channels])
        sino_T_E[:,:,i_energy] = cp.exp(-uL_E)
    print(f'raytacing  done, t={time() - t0:.2f}s')
        
    # process the transmitted energy information into an actual signal
    sino = detect_transmitted_sino(spec.E, spec.I0, sino_T_E, ct).get()
    print(f'forward project done, t={time() - t0:.2f}s')
    
    # fix div by 0?
    EPS = 1 #1e-8 
    sino[sino<EPS] = EPS
        
    return sino







