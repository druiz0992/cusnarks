# CUSNARKS

CUDA implementation of SNARKs proof using Groth protocol (based on https://github.com/iden3/snarkjs)


### Requirements 
* python (Tested with Python 2.7)
  - numpy
* cython : Cython will be used to build wrappers of C++/CUDA functions callable from Python modules
* c++/gcc
* CUDA toolkit

### Launch tests
make test

### Directory Structure
* build\    : Object files
* data\     : Auxiliary files
* lib\      : Generated dynamic libraries
* src\
  - cuda \     : C/C++/CUDA sources (.cpp, .c, .cu, .h)
  - cython \   : Cython files (.pyx, .pxd)
  - python \   : Python library (.py)
* test  \
  - python \ : Test source files (written in python)
