# CUSNARKS Overview
Optimized CUDA implementation of SNARK Groth16 prover based on [snarkjs][] designed with the objective of computing proofs for up
to 2^27 number of constraints in only few seconds. Cusnarks is expected to work with [circom][] for the generation and compilation
of circuits, and with [snarkjs][] for the computation of the trusted setup, witnesses and verification of the proof

Host side has been developed in C/C++/CUDA-C and Python. Python is the driving language where proof script is launched. Computation intensive functionality on the host side has been written in C/C++. Cython is used to build wrappers around C functions so that they can be
called from Python.

Elliptic curve scalar multiplication and reduction and polynomial multiplication, the heaviest functionality in terms of 
computation requirements has been implemented on the device side.

Two libraries are generated :
1. *libcusnarks.so* : is the standard cusnarks shared library
2. *pycusnarks.so* includes a cython wrapper so that it can be used from Python

## Outline
* [Architecture][]
* [Modules][]
* [Installation][]
* [Using Cusnarks][]
* [Other Info][]
  

## Architecture

Modules are divided into 4 categories depending on functionality:

1. **Infrastructure Layer** : Modules in this class have no dependencies and perform basic functionality commong to all project (constants and type defitiontion).  Infrastructure modules can be accessed by both host and device.

2. **Service Layer** : Modules in this class Implement non core functionality used by higher layerts (logging, random number generation or CUDA kernel launch abstraction). Service modules are implemented in C or CUDA C. All services layer modules are accessible by host side. Logging can be access by both host and device.

3. **Core Layer**  ; Modules in this category implement Snarks core functionalty,  including modular and elliptic curve implementation. Core functionality is duplicated in C and Python. Python was used as a fast prototyping implementation that could be used to validate C version, and not as an efficient Snarks implementation. C counter part on the other hand was designed with the main objective of being a very efficient implementation in terms of execution time. Thus, most modules are executed in the device side. Host side C core layemodules are mainly used to define kernel function handlers.

4. **Applications Layer** : User applications. For now only prover functionality is included, but in the future witness generation and trusted setup implemention will be in this layer. Application layer modules are implemented in Python and can launch CUDA kernels via cusnarks_kernel module and host side accelerated C functions via utils_host Cython wrapped module

![Architecture](doc/architecture.png)

## Modules

### Python

#### bigint.py

#### constants.py

#### ecc.py

#### groth_protocol.py

Diagram showing functionality and how everything is run 

#### zfield.py

#### z2field_element.py

#### zpoly.py

#### zutils.py

### C

#### constants.cpp

#### cuda.h

#### utils_host.cpp

#### utils_device.h

#### rng.cpp

#### types.h

### Cython
Function wrappers are defined in .pxd files and implemented in .pyx

#### cusnarks_kernel.pyx

### CUDA

#### cusnarks_kernel.cu

#### log.cu

#### ecbn128.cu / ecbn128_device.cu

#### ec2bn128.cu / ec2bn128_device.cu

#### u256.cu / u256_device.cu

#### zpoly.cu / zpoly_device.cu

#### z1_device.cu

#### z2_device.cu

## Data Types

## Installation
1. Download repository www.github.com/iden3/cusnarks.git

2. Ensure that all [dependencies][] are installed. 

3. Build libraries

```sh
make build
```
Libraries are stored in $CUSNARKS_HOME/lib

4. Launch units tests (optional)

```sh
make test
```
Launches all unit tests (python and C) 

5. Launch Groth prover instance

```sh
cd src/python
python groth_protocol.py
```

Launches the computation of a proof for a default circuit and witness

## Using-Cusnarks

### C

### Python


## Other
### Directory Structure
* *build\*    : Object files
* data\     : Auxiliary files (test circuits, precomputed roots of unity,...)
* lib\      : Generated dynamic libraries
* src\
  - cuda\     : C/C++/CUDA sources (.cpp, .c, .cu, .h)
  - cython\   : Cython files (.pyx, .pxd)
  - python\   : Python source files (.py)
* test\
  - python\   : Python unit test. They mainly test Python library using [unittest][] unit testing framework. 
   However, there are some files (*xxx_cu_xxx.py*) that test CUDA functions as well.
  - c\        : C unit tests for host side functionality.
  - ideas\    : Folder containing small scripts testing some ideas to be implmented in main code
* profiling\ : Profiling information
   python\   : Collection of scripts to measure time of CUDA functions 
* third_party_libs \ : Exteral libraries used will be automatically downloaded here
  - pcg-cpp  : implementation of PCG family of random number generators. Full details can be found at the [PCG-Random website].

### Requirements 
* python (Tested with Python 2.7)
  - numpy
* cython : Cython used to build wrappers of C++/CUDA functions callable from Python modules
* c++/gcc
* CUDA toolkit

[dependencies]: #Requirements 
[snarkjs]: https://www.github.com/iden3/snarkjs
[circom]: https://www.github.com/iden3/circom
[PCG-Random website]: http://www.pcg-random.org
[unittest]: https://python.org/3/library/unittest.html
[Architecture]: #Architecture
[Modules]: #Modules
[Installation]: #Installation
[Using Cusnarks]: #Using-Cusnarks
[Other Info]: #Other
