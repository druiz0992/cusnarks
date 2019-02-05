/*
    Copyright 2018 0kims association.

    This file is part of cusnarks.

    cusnarks is a free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    cusnarks is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along with
    cusnarks. If not, see <https://www.gnu.org/licenses/>.

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : bigint.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of biginteger class. 
// ------------------------------------------------------------------

*/

/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU
This class will get translated into python via swig
*/

#include <assert.h>
#include <iostream>

#include "types.h"
#include "bigint.h"
#include "bigint_device.h"

using namespace std;
/*
    Constructor

    Arguments :
      array_host : array of 256 bit numbers located at the host side
      length : Number of 256 bit numbers
*/
BigInt::BigInt (uint32_t* array_host, uint32_t *p, uint32_t length) {

  this->array_host = array_host;
  this->len = length;

  uint32_t size = len * sizeof(uint32_t) * NWORDS_256BIT;

  cudaError_t err = cudaMalloc((void**) &this->array_device, size);
  assert(err == 0);

  err = cudaMalloc((void**) &this->p, sizeof(uint32_t) * NWORDS_256BIT);
  assert(err == 0);

  err = cudaMemcpy(this->array_device, this->array_host, size, cudaMemcpyHostToDevice);
  assert(err == 0);

  err = cudaMemcpy(this->p, p, sizeof(uint32_t) * NWORDS_256BIT, cudaMemcpyHostToDevice);
  assert(err == 0);

}

void BigInt::addm() {
  //addm_kernel<<<64, 64>>>(array_device, p, len);
  addm_kernel(array_device, p, len);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

void BigInt::retrieve(uint32_t *array_host) {
  uint32_t size = len * sizeof(uint32_t) * NWORDS_256BIT ;
  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if(err != 0) { cout << err << endl; assert(0); }
}

BigInt::~BigInt() {
  cudaFree(array_device);
  cudaFree(p);
}
