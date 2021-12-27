# hydrogen-molecular-ion
### Introduction

This Jupyter Notebook is a guide through obtaining the molecular orbitals of the quantum system of H2+ by directly solving the Schrödinger equation, using Python (with modules such as NumPy and SciPy). This work is derived from the following resources.

- Grivet, J.-P. The Hydrogen Molecular Ion Revisited. Journal of Chemical Education, 2002, 79, 127. https://doi.org/10.1021/ed079p127.
- Johnson, J. L. Visualization of Wavefunctions of the Ionized Hydrogen Molecule. Journal of Chemical Education, 2004, 81, 1535. https://doi.org/10.1021/ed081p1535.1.

The sources should help when viewed alongside this Jupyter Notebook. The notations used are that of Grivet.

Prior knowledge of introductory quantum mechanics — particularly about the hydrogen atomic orbitals — is required. The following textbooks will be helpful with this regard.

For an in-depth review on quantum chemistry:

- *Quantum Chemistry* by Ira N. Levine

For an approachable yet complete review on quantum chemistry:

- *Physical Chemistry: A Molecular Approach* by McQuarrie and Simon

For an introduction to atomic orbitals and molecular orbitals:

- *Principles of Modern Chemistry* by David W. Oxtoby

As for the required background on mathematics, an elementary understanding of differential equations will suffice.

### Python information
The following modules are *required*.

- NumPy
- SciPy for solving differential equations
- Matplotlib for 2D graphing

Other modules used are

- Plotly for 3D graphing (required for viewing 3D plots, images of which are already generated)
- scikit-image (for generating 3D meshes, which are already generated)