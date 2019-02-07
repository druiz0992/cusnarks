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
//  Implementation of CUDA biginteger class. It provides the functionality
//    to perform arithmetic operations modulo p on a given vector. 
//   
// TODO
//    - Move modulo to constant memory
//    - Use managed memory for data
// ------------------------------------------------------------------

*/

#include <assert.h>
#include <iostream>

#include "types.h"
#include "bigint.h"
#include "bigint_device.h"

using namespace std;

/*
    Constructor : Reserves device memory for vector and modulo p. 

    Arguments :
      p : 256 bit number in 8 word uint32 array
      length : Vector length for future arithmetic operations
*/
BigInt::BigInt (const uint32_t *p, const uint32_t device_vector_len) : in_vector_len(device_vector_len)
{
  uint32_t size = in_vector_len * sizeof(uint32_t) * NWORDS_256BIT;

  // Allocate global memory in device for input and output
  cudaError_t err = cudaMalloc((void**) &this->in_vector_device, size);
  assert(err == 0);

  err = cudaMalloc((void**) &this->out_vector_device, size/2);
  assert(err == 0);

  // Allocate global memory for modulo p
  err = cudaMalloc((void**) &this->p, sizeof(uint32_t) * NWORDS_256BIT);
  assert(err == 0);

  // Copy modulo p to device memory
  err = cudaMemcpy(this->p, p, sizeof(uint32_t) * NWORDS_256BIT, cudaMemcpyHostToDevice);
  assert(err == 0);
}

/*
    Modular addition. 

    Arguments :
      in_vector_host : Input vector of upto N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector_host : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      len : number of elements in input vector. Cannot be greater than amount reseved during constructor
*/
void BigInt::addm(uint32_t *in_vector_host, uint32_t *out_vector_host, uint32_t len)
{
  if (len > in_vector_len) { return; }

  copyVectorToDevice(in_vector_host, len);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = 256;
  gridD = (len + blockD - 1) / blockD;
  addm_kernel<<<gridD, blockD>>>(in_vector_device, p, in_vector_len, out_vector_device);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);

  cudaDeviceSynchronize();

  copyVectorFromDevice(out_vector_host, len/2);
}

/*
    Transfer input vector from host to device

    Arguments :
      in_vector_host : Input vector of upto N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      len : number of elements in input vector to be xferred. 
          Cannot be greater than amount reseved during constructor, but not checked
*/
void BigInt::copyVectorToDevice(uint32_t *in_vector_host, uint32_t len)
{
  uint32_t size = len * sizeof(uint32_t) * NWORDS_256BIT ;
  // Copy input data to device memory
  cudaError_t err = cudaMemcpy(in_vector_device, in_vector_host, size, cudaMemcpyHostToDevice);
  assert(err == 0);
}

/*
    Transfer output vector from device to host

    Arguments :
      out_vector_host : Output vector of upto N/2 256 bit elements Y[0], Y[1], Y[2] ... Y[N/2-1].
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
void BigInt::copyVectorFromDevice(uint32_t *out_vector_host, uint32_t len)
{
  uint32_t size = len * sizeof(uint32_t) * NWORDS_256BIT ;
  
  // copy results from device to host
  cudaMemcpy(out_vector_host, out_vector_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

BigInt::~BigInt()
{
  cudaFree(in_vector_device);
  cudaFree(out_vector_device);
  cudaFree(p);
}
