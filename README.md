# dex-ct-sim
Dual-energy CT raytracing simulator with basis material decomposition.
These scripts are for:
1. Generating single-energy CT sinograms using Siddons raytracing
2. Reconstructing CT images using fan-beam filtered back-projection
3. Generating dual-energy CT basis material sinograms using Gauss-Newton
   decomposition and reconstructing basis material images using FFBP.



## main.py
Main file for running the dual-energy CT simulation. There are three classes 
of parameters that must be defined by the user:
- ScannerGeometry: the CT imaging geometry. Include the number of channels, projections, fan angle, source-isocenter-distance (SID), source-detector-distance (SDD), and pixel shape.
- Phantom: the voxelized object to be imaged. Include the phantom filename, phantom name, atomic composition filename, and number/size [cm] of pixels in x, y, z directions.
- Spectrum: the polyenergetic x-ray spectrum for a single-energy CT acquisition. Include the filename and target dose [Gy].

In addition, three image reconstruction parameters should be assigned:
- N_matrix: reconstructed image matrix size
- FOV: reconstructed field-of-view [cm]
- ramp: reconstruction filter cutoff percentage of Nyquist frequency


## system.py
Definitions of the three classes to set up each simulation. 

## forward_project.py
Functions for forward projecting through the phantom to create a sinogram. Utilizes Siddon's algorithm.<sup>1</sup>

## back_project.py
Functions for back projecting a sinogram to create a reconstructed image. Utilizes fan-beam filtered back projection with a sinc window filter.

## matdecomp.py
Functions for performing basis material decomposition using two sinograms acquired with different polychromatic x-ray spectra. Utilizes a Gauss-Newton algorithm.<sup>2</sup>

## plots.py
Script used to generate plots to analyze data output from main.

  
  
<sup>1</sup>Siddon, Robert L. "Fast calculation of the exact radiological path for a three‐dimensional CT array." Medical physics 12, no. 2 (1985): 252-255.

<sup>2</sup>D. Rigie and P. J. La Riviere, “An efficient spectral CT material decomposition method using the Gauss–Newton algorithm,” Proc. 2015 IEEE Medical Imaging Conference (2015).
