 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# 2D Acoustic Wave Propagation
## acoustic wave propagation in frequency domain using 13 and 5 point stencils

The wave propagation modeling is created using the fourth- and second-order staggered-grid finite difference approximation of the scalar wave equation resulting in 13- and 5-point stencils. The modeling is pperformed in frequency domain for dicreet frequencies. For stability and for avoiding numerical error, there should be at least 4 (13-point stencil) and 10 (5-point stencil) gridpoints per minimum wavelength. The left, right and bottom boundaries are incorporated with Perfectly Matching Layers (PMLs) to suppress undesired reflections from edges. The top boundary is left as free surface.

All the required functions are in the file ***functions.py***

## Reference

Bernhard Hustedt, Stéphane Operto, Jean Virieux, Mixed-grid and staggered-grid finite-difference methods for frequency-domain acoustic wave modelling, Geophysical Journal International, Volume 157, Issue 3, June 2004, Pages 1269–1296, https://doi.org/10.1111/j.1365-246X.2004.02289.x
