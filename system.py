#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 13:31:59 2022

@author: gjadick
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat

import xcompy as xc


def get_matcomp_dict(filename):
    '''
    Create dictionary of material compositions.
    The phantom file is made of number IDs that correspond to a density [g/cm3]
    and atomic composition by weight, which must be stored for getting x-ray
    attenuation coefficients later on.
    '''
    f = open(filename, 'r')
    L_raw = [l.strip() for l in f.readlines() if len(l.strip())]
    f.close()
    
    # split each line
    mat_dict = {}
    header = L_raw[0].split()
    for line in L_raw[1:]:
        
        split = line.split()
        N, density = int(split[0]), float(split[2])
        
        mat_dict[N] = [density, '']
        for i,val in enumerate(split[3:]):
            mat_dict[N][1] += f'{header[3+i]}({val})'

    return mat_dict


class Phantom:
    '''
    Class to handle a 3D voxelized phantom and corresponding params.
    '''
    def __init__(self, filename, phantom_id, matcomp, Nx, Ny, Nz, 
                 sx=0.1, sy=0.1, sz=0.1, ind=0, dtype=np.uint8):
        self.filename = filename
        self.name = phantom_id
        
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
        
        self.matcomp_filename = matcomp
        
        # get the linear attenuation coefficients for the spectrum energies
        self.matcomp_dict = get_matcomp_dict(self.matcomp_filename)
        self.matkeys = np.array(list(self.matcomp_dict.keys()), dtype=dtype)
        
        # init other vars
        self.u_dict = None
        self.u_arr = None
        self.M3D_seq = None
        self.M_seq = None
        
        def get_u_dict(self, energies):
            u_dict = {}
            for mat in self.matcomp_dict:
                density, matcomp = self.matcomp_dict[mat] #[1.0, 'H(11.2)O(88.8)'] ex: water
                u_E = density * xc.mixatten(matcomp, energies)
                u_E[np.isnan(u_E)] = 0.0
                u_dict[mat] = u_E
            self.u_dict = u_dict
            return u_dict
        
        def get_u_arr(self, energies):
            N = len(energies)            # num energies
            M = len(self.matkeys)     # num materials
            u_dict = get_u_dict(self, energies)
            u_arr = np.zeros(N*M, dtype=np.float32)  # initialize array
            for m in np.array(self.matkeys, dtype=int).sort():
                u_arr[m*N: (m+1)*N] = u_dict[m]
            self.u_arr = u_arr
            return u_arr
        
    def M_mono(self, E0, HU=True):
        matcomp_dict = get_matcomp_dict(self.matcomp_filename)
        u_dict = {}
        for mat in matcomp_dict:
            density, matcomp = matcomp_dict[mat] 
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
        
        
class ScannerGeometry:
    def __init__(self, SID=50.0, SDD=100.0, N_channels=360, gamma_fan=np.pi/4, N_proj=1000, theta_tot=2*np.pi, pxshape='rect'):

        # name for this geometry        
        self.geo_id = f'{int(SID)}cm_{int(SDD)}cm_{int(180*gamma_fan/np.pi)}fan_{N_proj}view_{N_channels}col'     

        # source-isocenter and source-detector distances
        self.SID = SID
        self.SDD = SDD
        
        # test detector, multi-channel
        self.N_channels = N_channels
        self.gamma_fan = gamma_fan
        self.dgamma_channel = gamma_fan/N_channels
        if N_channels%2==1:        # odd
            self.gammas = np.arange(-gamma_fan/2 + self.dgamma_channel/2, gamma_fan/2, self.dgamma_channel)    
        else:
            self.gammas = np.arange(-gamma_fan/2, gamma_fan/2, self.dgamma_channel)    

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
    def __init__(self, filename, name, mono_E=False):
        
        # 1 uGy dose at 20cm water depth, rescale constants.
        # These were computed with https://github.com/gjadick/dex-single-ray
        rescale_1uGy_dict = {
            '6MV':       271.5478714123602,
            'detunedMV': 1.426161029392114,
            '80kV':      781.7141362326054,
            '120kV':     345.57004865705323,
            '140kV':     267.0880507831676}
            
        # Effective mu_water and mu_air dictionaries for HU conversions,
        # found by simulating noiseless images of water phantom with each 
        # spectrum and measuring the mu value in its center.
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
        E_eff_dict = {  # linear interp of u_water_dict with NIST curve
            '6MV':       2692.36  ,
            'detunedMV': 1753.73  ,
            '80kV':      46.32  ,
            '120kV':     57.91, 
            '140kV':     63.79    }
        
        # measured from 150 proj x 100 col, 1 mGy scan for HU conversion
        self.u_water = u_water_dict[name]
        self.u_air = u_air_dict[name]
        self.E_eff = E_eff_dict[name]
        
        self.filename = filename 
        self.name = name
        
        # attempt reading file
        try:  
            if 'Accuray' in filename:
                spec_data = pd.read_csv(filename, sep=',')
                self.I0_raw = spec_data['Weight'].to_numpy()
                self.E = spec_data['MeV'].to_numpy()*1000 # convert MeV -> keV
            else:
                spec_data = loadmat(filename)
                self.I0_raw = spec_data['ss'][:,0]
                self.E = spec_data['ee'][:,0]
        except:
            print(f"Failed to open spectrum filename {filename}, failed to initialize.")
        
        if mono_E:
            print('Debugging! Monoenergetic on! 80kV')
            self.E = np.array([80.0])
            self.I0_raw = np.array([1.0e8]) # arbitrary counts
            self.name = 'mono80keV'
            
        # scale counts to 1 mGy dose at 20 cm water depth
        self.I0 = self.I0_raw * rescale_1uGy_dict[name] * 1e3 # uGy -> mGy
        
        # include energies rescaled to 1 keV bins
        self.E_1keV = np.arange(1.0, np.max(self.E)+1, 1.0)
        self.I0_1keV = np.interp(self.E_1keV, self.E, self.I0)

    def get_counts(self):
        return np.trapz(self.I0, x=self.E)
    
    def rescale_I0(self, scale, verbose=False):
        if verbose:
            print(f'rescaled, {self.get_counts():.2e} -> {scale*self.get_counts():.2e}')
        self.I0 = self.I0 * scale
        self.I0_1keV = self.I0_1keV * scale




    