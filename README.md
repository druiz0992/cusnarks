# CUSNARKS

CUDA implementation of SNARKs proof Groth protocol (based on https://github.com/iden3/snarkjs)


### Requirements 
* python (Tested with Python 2.7)
  - numpy
* cython : Cython will be used to build wrappers of C++/CUDA functions callable from Python modules
* c++/gcc
* CUDA toolkit

### Directory Structure
* build\    : Object files
* data\     : Auxiliary files
* lib\      : Generated dynamic libraries
* src\
  - c \     : C/C++ sources (.cpp, .c, .cu)
  - include : Header files (.h)
  - c-wrappers : Cython files (.pyx, .pxd)
* test  \
  - python \ : Test source files (written in python)
