#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 13:31:59 2022

@author: gjadick
"""
import numpy as np
import cupy as cp
from scipy.io import loadmat
import pandas as pd 
import xcompy as xc


class Phantom:
    '''
    Class to handle a 3D voxelized phantom and corresponding params.
    
    phantom_id -- name for the phantom (used for saving outputs)
    filename -- name of the binary file containing the phantom data. 
                should be formatted as a raveled array of integers, 
                each of which corresponds to a material with elemental
                composition and density listed in the matcomp file
    matcomp_filename -- file with the material compositions corresponding 
                        to each integer of the phantom file. Should be 
                        formatted with four tab-separated columns: ID, 
                        material name, density, elemental composition by 
                        weight (see example file)
    Nx, Ny, Nz -- shape of the phantom in 3 dimensions (if 2D, set Nz=1)
                  The x-y plane corresponds to each axial image.
    ind -- z-index of the phantom slice to use if 3D (if 2D, leave ind=0)
    sx, sy, sz -- size of each voxel in cm
    dtype -- data type for raveled phantom file (default uint8)
    '''
    def __init__(self, phantom_id, filename, matcomp_filename, Nx, Ny, Nz, 
                 sx=0.1, sy=0.1, sz=0.1, ind=0, dtype=np.uint8):

        self.name = phantom_id
        self.filename = filename
        self.matcomp_filename = matcomp_filename

        # number of voxels in each dimension
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        
        # size of voxels in each dimension [cm]
        self.sx = sx
        self.sy = sy
        self.sz = sz
        
        # read the 3D phantom
        # each uint8 value should correspond to some material
        self.M3D = np.fromfile(filename, dtype=dtype).reshape([Nz, Ny, Nx])
        
        # assign sample image
        self.ind = ind
        self.M = self.M3D[ind]        
        
        # get the linear attenuation coefficients for the spectrum energies
        self.matcomp_dict = get_matcomp_dict(self.matcomp_filename)
        self.matkeys = np.array(list(self.matcomp_dict.keys()), dtype=dtype)

    def M_mono_stack(self, energies):
        N_energies = len(energies)
        M_mono_stack = cp.zeros([N_energies, self.Nx, self.Ny], dtype=cp.float32)        
        for i_mat in self.matkeys:
            density, matcomp = self.matcomp_dict[i_mat] 
            u_E = density * cp.array(xc.mixatten(matcomp, energies), dtype=cp.float32)
            for i_E in range(N_energies):
                cp.place(M_mono_stack[i_E], self.M==i_mat, u_E[i_E])
        return M_mono_stack        
    
    def M_mono(self, E0, HU=True):  # !!! might want to combine with M_mono_stack for speed
        u_dict = {}
        for mat in self.matcomp_dict:
            density, matcomp = self.matcomp_dict[mat] 
            u_E = density * float(xc.mixatten(matcomp, np.array([E0], dtype=np.float64)))
            u_dict[mat] = u_E
        M_mono = np.zeros(self.M.shape, dtype=np.float32)
        for i in range(self.Nx):
            for j in range(self.Ny):
                M_mono[i,j] = u_dict[self.M[i,j]]
        if HU:  # convert attenuation to HU 
            u_w = float(xc.mixatten('H(11.2)O(88.8)', np.array([E0], dtype=np.float64)))
            M_mono = 1000*(M_mono-u_w)/u_w
        return M_mono


def get_matcomp_dict(filename):
    '''
    Convert material composition file into a dictionary of density/matcomp strings.
    '''
    with open(filename, 'r') as f:
        L_raw = [l.strip() for l in f.readlines() if len(l.strip())]
    mat_dict = {}
    header = L_raw[0].split()
    for line in L_raw[1:]:
        split = line.split()  # separate into four columns
        N    = int(split[0])
        name = split[1]
        density = float(split[2])
        matcomp = split[3]        
        mat_dict[N] = [density, matcomp]  # add dictionary entry
    return mat_dict


class ScannerGeometry:
    def __init__(self, eid=True, detector_file=None, pxshape='rect',
                 SID=50.0, SDD=100.0, N_channels=360, gamma_fan=np.pi/4, 
                 N_proj=1000, theta_tot=2*np.pi):

        self.eid = eid  
        if eid: 
            self.det_mode = 'eid'  # energy integrating
        else:
            self.det_mode = 'pcd'  # photon counting
            
        if detector_file is None:  # ideal detector?
            self.det_E = [1.0]
            self.det_eta_E = [1.0]
        else:
            data = np.fromfile(detector_file, dtype=np.float32)
            N_det_energy = len(data)//2
            self.det_E = data[:N_det_energy]      # 1st half is energies
            self.det_eta_E = data[N_det_energy:]  # 2nd half is detective efficiencies
 
        # name the geometry        
        #self.geo_id = f'{int(SID)}cm_{int(SDD)}cm_{int(180*gamma_fan/np.pi)}fan_{N_proj}view_{N_channels}col'     
        self.geo_id = f'{int(SID)}cm_{int(SDD)}cm_{int(180*gamma_fan/np.pi)}fan_{N_proj}view_{N_channels}col_{self.det_mode}'
        
        # source-isocenter and source-detector distances
        self.SID = SID
        self.SDD = SDD
        
        # test detector, multi-channel
        self.N_channels = N_channels
        self.gamma_fan = gamma_fan
        self.dgamma_channel = gamma_fan/N_channels
        # !!! gammas !!!
        #if N_channels%2==1: 
        #    self.gammas = np.arange(-gamma_fan/2 + self.dgamma_channel/2, gamma_fan/2, self.dgamma_channel)    
        #else:
        #    self.gammas = np.arange(-gamma_fan/2, gamma_fan/2, self.dgamma_channel)    
        self.gammas = np.arange(-gamma_fan/2, gamma_fan/2, self.dgamma_channel) + self.dgamma_channel/2

        # sampling distance, detector plane
        self.s = SDD*gamma_fan/N_channels
        
        # sampling distance, isocenter
        self.s_iso = SID*gamma_fan/N_channels
        if pxshape=='square':
            self.h_iso = self.s_iso
        else:
            self.h_iso = 1.0  # 10 mm height at iso 
        self.A_iso = self.s_iso*self.h_iso   # detector pixel area at isocenter, assume square
        
        # projections
        self.N_proj = N_proj
        self.theta_tot = theta_tot
        self.dtheta_proj = theta_tot/N_proj
        self.thetas = np.arange(0, theta_tot, self.dtheta_proj )
        
        # just a check
        if len(self.thetas) > N_proj:
            self.thetas = self.thetas[:N_proj]

        if len(self.gammas) > N_channels:
            self.gammas = self.gammas[:N_channels]


    
class Spectrum:
    def __init__(self, filename, name, mono_E=None):
            
        # Effective mu_water and mu_air dictionaries for HU conversions,
        # found by simulating noiseless images of water phantom with each 
        # spectrum and measuring the mu value in its center (150 projections, 
        # 100 detector channels, 1 mGy dose).
        # These might be affected by beam hardening.
        u_water_dict = {
            '6MV':       0.04268331080675125  ,
            'detunedMV': 0.05338745564222336  ,
            '80kV':      0.24212932586669922  ,
            '120kV':     0.21030768752098083  ,
            '140kV':     0.2016972303390503  }
        u_air_dict = {
            '6MV':       0.00024707260308787227  ,
            'detunedMV': 0.00031386411865241826  ,
            '80kV':      0.002364289714023471  ,
            '120kV':     0.0016269732732325792  ,
            '140kV':     0.0014648198848590255  }
        E_eff_dict = {  # linear interp of u_water_dict with NIST curve [keV]
            '6MV':       2692.36  ,
            'detunedMV': 1753.73  ,
            '80kV':      46.32  ,
            '120kV':     57.91, 
            '140kV':     63.79    }
        
        self.filename = filename 
        self.name = name
        try:  
            data = np.fromfile(filename, dtype=np.float32)
            self.E, self.I0 = data.reshape([2, data.size//2])
            self.u_water = u_water_dict[name]
            self.u_air = u_air_dict[name]
            self.E_eff = E_eff_dict[name]
        except:
            print(f"Failed to open spectrum filename {filename}, failed to initialize.")
        
        # For debugging, can use a monoenergetic x-ray beam.
        if mono_E is not None:
            print(f'Debugging! Monoenergetic on! {mono_E} keV')
            self.E = np.array([mono_E])
            self.I0_raw = np.array([1.0e8]) # arbitrary counts
            self.name = f'mono{mono_E:04}keV'

    def get_counts(self):
        return np.trapz(self.I0, x=self.E)
    
    def rescale_I0(self, scale, verbose=False):
        print(f'rescaled counts : {self.get_counts():.2e} -> {scale*self.get_counts():.2e}')
        self.I0 = self.I0 * scale









    
    
    
    
    