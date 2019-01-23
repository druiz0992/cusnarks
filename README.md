# CUSNARKS

CUDA implementation of SNARKs proof Groth protocol (based on https://github.com/iden3/snarkjs)

### Current Milestone
Implementation of the protocol in Python. Required modules:
* Big Integer Library : Done
* Finite Field Library : Done
** Montgomery reduction
* Elliptical Curve : Done
** Points can be defined in Affine, Projective or Jacobian coordinates : Pending Jacobian implementation, but left for next stage once we can evaluate performance
** Finite field elements can be defined in default or Montfomery format : Done
* Poly : Ongoing
** Arithmetic written, pending sparse polys, tests ongoing.
* Groth protocol : Pending

### TODO
* Profiling : Measure times to evaluate improvements with CUDA implementation. 
* Wrappers  : Define interface between Python and C++. Preliminary approach is to use cython.
* Module optimization : Finite Field arithmetic, Elliptical curve arithmetic and Polynomial arithmetic are
     the main targets for optimization. Special care needs to be given to amount of (>8GB) data that needs to
     be processed. How memory is managed and accessed between CPU and GPU domain will be critical
* Data interface : Define how input data (what do we want to proof) is passed to CUSNARKs. For proof of concept 
  we will use json. Once full chain is working, something else will be defined that is capable of handling large
  amounts of data

### Requirements 
* Pthon (Tested with Python 2.7)
**  numpy
* Cython : Cython will be used to build wrappers of C++/CUDA functions callable from Python modules
* c++/gcc
* CUDA toolkit

### Directory Structure
* build\    : Object files
* data\     : Auxiliary files
* lib\      : Generated dynamic libraries
* src\
**  c \     : C/C++ sources (.cpp, .c, .cu)
**  include : Header files (.h)
**  c-wrappers : Cython files (.pyx, .pxd)
* test  \
**  python \ : Test source files (written in python)
