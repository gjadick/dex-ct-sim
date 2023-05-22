#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:03:45 2022

@author: gjadick
"""

import numpy as np
from time import time
    
def pre_process(sino_log, ct, ramp_percent):
    ''' pre-process the projections for recon
    applies fan-beam filter and sinc window for noise suppression
    '''
    
    # modified fanbeam data (q --> qm)
    sino_qm = np.array([q*ct.SID*np.cos(ct.gammas) for q in sino_log], dtype=np.float32)    # with cosine weighting
         
    # # Hsieh method for fanbeam ramp filter
    g = np.zeros(ct.N_channels)
    n = np.arange(-ct.N_channels//2, ct.N_channels)
    
    if ct.N_proj%2==1:   # this is buggy :-( 
        n = np.arange(-ct.N_channels//2+1, ct.N_channels)
    else:
        n = np.arange(-ct.N_channels//2, ct.N_channels)
    
    for i in range(ct.N_channels):
        if n[i]==0:
            g[i] = 1/(8 * ct.dgamma_channel**2)
        elif n[i]%2==1: # odd
            g[i] = -0.5/(np.pi * np.sin(ct.gammas[i]))**2

    # implement sinc window (Hsieh CT 4th edition, eq. 3.31 )
    G = np.fft.fft(g)
    w = np.fft.fftfreq(g.size)  # frequencies
    window = np.zeros(ct.N_channels)
    wL, wH = 0, ramp_percent*np.max(w)
    for i in range(ct.N_channels):
        if np.abs(w[i]) <= wL:
            window[i] = 1.0
        elif wL < np.abs(w[i]) and np.abs(w[i]) <= wH:
            frac = np.pi*(np.abs(w[i])-wL)/(wH-wL)
            window[i] = np.sin(frac)/frac
            
    g = np.real(np.fft.ifft(G*window))

    # convolve and scale by dgamma
    sino_filtered = np.array([np.convolve(qm, g, mode='same') for qm in sino_qm], dtype=np.float32)        
    sino_filtered *= ct.dgamma_channel
    
    return sino_filtered
        

def get_angle(x,y):
    ''' calcs angle in x-y plane, shifts to range (0, 2pi)'''
    theta = np.arctan(y/x)
    if x < 0:  # quadrants 2,3
        theta += np.pi
    elif y < 0: # quadrant 4
        theta += 2*np.pi
    return theta    


def get_L2(r, theta, beta, SID):
    '''calcs rescale value for projection'''
    L2 = r**2 * np.cos(beta - theta)**2 + (SID + r*np.sin(beta - theta))**2
    return np.float32(L2)


def get_gamma(r, theta, beta, SID):
    '''calcs the gamma passing through given matrix (x,y) coordinate '''
    gamma = np.arctan(r*np.cos(beta - theta)/(SID + r*np.sin(beta - theta)))
    return np.float32(gamma)


def get_recon_coords(N_matrix, FOV, N_proj_rot, dbeta_proj, SID):
    '''Get the coordinates needed for the reconstruction matrix (common to all recons with same matrix)'''
    ## matrix coordinates: (r, theta)
    sz = FOV/N_matrix 
    matrix_coord_1d = np.arange((1-N_matrix)*sz/2, N_matrix*sz/2, sz)
    X_matrix, Y_matrix = np.meshgrid(matrix_coord_1d, -matrix_coord_1d)
    r_matrix = np.sqrt(X_matrix**2 + Y_matrix**2)
    theta_matrix = np.reshape([get_angle(X_matrix.ravel()[i], Y_matrix.ravel()[i]) for i in range(N_matrix**2)], [N_matrix, N_matrix])
    r_matrix, theta_matrix = np.float32(r_matrix), np.float32(theta_matrix)

    # fan-beam rescaling for each beta: L^2(r,theta), gamma(r,theta)
    gamma_target_matrix_all = np.empty([N_proj_rot, N_matrix, N_matrix], dtype=np.float32)
    L2_matrix_all = np.empty([N_proj_rot, N_matrix, N_matrix], dtype=np.float32)
    for i_beta in range(N_proj_rot):
        beta = i_beta*dbeta_proj            
        gamma_target_matrix_all[i_beta] = get_gamma(r_matrix, theta_matrix, beta, SID)
        L2_matrix_all[i_beta] = get_L2(r_matrix, theta_matrix, beta, SID)

    # recon matrix indices
    ji_coord = np.reshape(np.array(np.meshgrid(range(N_matrix),range(N_matrix))).T, [N_matrix**2, 2])

    # get rid of coordinates outside of max FOV
    ji_mask = [ np.sqrt((j+0.5-N_matrix/2)**2 + (i+0.5-N_matrix/2)**2) < N_matrix/2 for j,i in ji_coord]
    ji_coord = ji_coord[ji_mask]

    return ji_coord, r_matrix, theta_matrix, gamma_target_matrix_all, L2_matrix_all
    

def lerp(v0, v1, t):
    '''linear interp'''
    return (1-t)*v0 + t*v1


def do_recon(sinogram, dbeta_proj, gamma_coord,                  
             gamma_target_matrix_all, L2_matrix_all, ji_coord,
             verbose=False, verbose_plots=False):
    '''
    Main reconstruction program. Reconstructs the sinogram.
    Parameters
    ----------
    sinogram : 2D matrix
        Pre-processed sinogram for reconstruction.
    dbeta_proj : float
        Change in beta angle for each projection [rad].
    gamma_coord : 1D array
        Local angle coordinate for each column [rad].
    gamma_target_matrix_all : 3D array
        gamma targets for linear interpolation for each (i,j,beta)
    L2_matrix_all : 3D array
        L^2 normalization factors for each (i,j,beta)
    ji_coord : 2D array
        List of the [j,i] coordinates (corresponding to y, x in recon matrix)
    verbose : bool, optional
        whether to print timing. The default is False.
    Returns
    -------
    2D matrix
        the reconstruction.
    '''
    t0 = time()

    matrix = np.zeros(L2_matrix_all[0].shape)
    
    for i_beta in range(len(sinogram)):
        proj_z = sinogram[i_beta] # fan-beam data at this z
        
        if verbose:
            if i_beta%100 == 0:
                print(f' {100*i_beta/len(sinogram):5.1f}%: {time()-t0:.3f} s')
        
        L2_matrix = L2_matrix_all[i_beta]         # matrix with L^2 factors
        gamma_max  = np.max(gamma_coord)
        dgamma = gamma_coord[1]-gamma_coord[0]
        for j,i  in ji_coord:
            gamma_target = gamma_target_matrix_all[i_beta,j,i]
            if np.abs(gamma_target) >= gamma_max:
                pass
            else:
                i_gamma0 = int((gamma_target + gamma_max)//dgamma)
                t = (dgamma*(i_gamma0+1) - gamma_max - gamma_target)/dgamma
                this_q = lerp(proj_z[i_gamma0], proj_z[i_gamma0 + 1], t)
                matrix[j,i] += this_q * dbeta_proj  / L2_matrix[j,i]  
           
    return matrix.T


def do_recon_gpu(sino, gamma_target_M, L2_M, gamma_coord, dbeta_proj):
    
    import pycuda.autoinit
    from pycuda import compiler, driver, gpuarray

    N_proj, cols = sino.shape
    N_matrix = L2_M.shape[1]
    gamma_max = np.max(gamma_coord)
    dgamma = gamma_coord[1]-gamma_coord[0]

    # block/thread allocation warning
    block_max=1024
    
    block_gpu=(N_matrix, block_max//N_matrix, 1)
    grid_gpu=(1,N_matrix//(block_max//N_matrix))
        
    if N_matrix > block_max:
        print(f'need to manually set GPU block/thread for large matrix {N_matrix} > {block_max}')
    
    if np.log2(N_matrix)%1!=0:
        print(f'may need to manually set GPU block/thread for {N_matrix} size (not power of 2)')
        # bandaid for small matrix! This will not work larger than block_max
        block_gpu=(N_matrix, 1, 1)
        grid_gpu=(1, N_matrix)

    #print('GPU block', block_gpu)
    #print('GPU grid', grid_gpu)
    
    kernel_code_template = """
        #include <math.h>
    
        __global__ void do_recon(float *matrix, float *sino,  float *gamma_target_M, float *L2_M) {
            
            // get i, j for the matrix coordinate
            int i = threadIdx.x + blockDim.x * blockIdx.x;
            int j = threadIdx.y + blockDim.y * blockIdx.y;
       
            // result at pixel (i,j)
            float result = 0.0;
            

            // assign constants
            int N_proj = %(N_PROJ)s;
            int N_matrix = %(N_MATRIX)s;
            int N_cols = %(COLS)s;
            float gamma_max = %(GAMMA_MAX)s; 
            float dgamma = %(DGAMMA)s;
            float dbeta_proj = %(DBETA_PROJ)s;

        
            // check if pixel within FOV, otherwise skip
            //if( sqrt( powf( (float)i + 0.5 - (float)N_matrix/2 , 2) +  powf( (float)j + 0.5 - (float)N_matrix/2 , 2) ) < (float)N_matrix/2 ) {
            if(true) {
                for(int i_beta=0; i_beta < N_proj; i_beta++) {
                    float L2 =           L2_M          [ i_beta*N_matrix*N_matrix + j*N_matrix + i ];
                    float gamma_target = gamma_target_M[ i_beta*N_matrix*N_matrix + j*N_matrix + i ];
            
                    if(fabsf(gamma_target) <  gamma_max) { 
                        int i_gamma0 = (int)((gamma_target + gamma_max)/dgamma);
                        float t = (dgamma*(i_gamma0+1) - gamma_max - gamma_target)/dgamma;
                    
                        // linear interp
                        float this_q = (1-t)*sino[ i_beta*N_cols + i_gamma0] +   t*sino[ i_beta*N_cols  + i_gamma0 + 1];
                
                        // add to results
                        result = result + (this_q * dbeta_proj / L2);
                    }
                }
            }
            // write result to matrix
            matrix[ j * N_matrix + i ] = result;
        }
    """

    kernel_code = kernel_code_template % {
            'N_MATRIX':   N_matrix,
            'N_PROJ':     N_proj,
            'GAMMA_MAX':  gamma_max,
            'DGAMMA':     dgamma,
            'DBETA_PROJ': dbeta_proj,
            'COLS':       cols
            }

    # compile code
    mod = compiler.SourceModule(kernel_code)

    # get kernel function from compiled code
    do_recon_gpu = mod.get_function("do_recon")

    # move stuff to GPU
    sino = gpuarray.to_gpu(sino)
    gamma_target_M = gpuarray.to_gpu(gamma_target_M)
    L2_M = gpuarray.to_gpu(L2_M)

    # do the recon
    matrix = gpuarray.empty([N_matrix, N_matrix], np.float32)
    do_recon_gpu(matrix, sino, gamma_target_M, L2_M, 
                 block=block_gpu, grid=grid_gpu)

    return matrix.get()


