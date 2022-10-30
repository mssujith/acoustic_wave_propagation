# 2D acoustic_wave_propagation
## 2D acoustic wave propagation in frequency domain using 13 and 5 point stencils

The wave propagation modeling is created using the fourth-order staggered-grid finite difference approximation (13-point stencils) of the scalar wave equation in frequency domain. For stability and avoid numerical error, there should be at least 4 gridpoints per minimum wavelength. The boundaries are incorporated with Perfectly Matching Layers (PMLs) to suppress undesired reflections from edges.

All the required functions are in the file **functions.py**

## Reference

Bernhard Hustedt, Stéphane Operto, Jean Virieux, Mixed-grid and staggered-grid finite-difference methods for frequency-domain acoustic wave modelling, Geophysical Journal International, Volume 157, Issue 3, June 2004, Pages 1269–1296, https://doi.org/10.1111/j.1365-246X.2004.02289.x
