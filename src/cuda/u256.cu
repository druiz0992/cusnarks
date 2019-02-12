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
// File name  : u256.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of CUDA 256 bit unsigned integer class. It provides the functionality
//    to perform arithmetic operations
//   
// TODO
//    - Move modulo to constant memory
//    - Use managed memory for data
// ------------------------------------------------------------------

*/

#include <assert.h>
#include <iostream>
#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "rng.h"
#include "u256.h"
#include "u256_device.h"

#define U256_BLOCK_DIM  (256)

using namespace std;

/*
    Constructor : Reserves device memory for vector and modulo p. 

    Arguments :
      p : 256 bit number in 8 word uint32 array
      length : Vector length for future arithmetic operations
*/
U256::U256 (const uint32_t *p, const uint32_t device_vector_len) : in_vector_len(device_vector_len)
{
  U256(p, device_vector_len, 0);
}

U256::U256 (const uint32_t *p, const uint32_t device_vector_len, const uint32_t seed) : in_vector_len(device_vector_len)
{
  uint32_t size = in_vector_len * sizeof(uint32_t) * NWORDS_256BIT;

  // Allocate global memory in device for input and output
  CCHECK(cudaMalloc((void**) &this->in_vector_device, size));

  CCHECK(cudaMalloc((void**) &this->out_vector_device, size/2));

  // Allocate global memory for modulo p
  CCHECK(cudaMalloc((void**) &this->p, sizeof(uint32_t) * NWORDS_256BIT));

  // Copy modulo p to device memory
  CCHECK(cudaMemcpy(this->p, p, sizeof(uint32_t) * NWORDS_256BIT, cudaMemcpyHostToDevice));

  if (seed == 0){ rng =  _RNG::get_instance(); }
  else { rng = _RNG::get_instance(seed); }
}

void U256::rand(uint32_t *samples, const uint32_t n_samples)
{
    rng->randu32(samples, n_samples * NWORDS_256BIT);
}

/*
    Modular addition. 

    Arguments :
      in_vector_host : Input vector of upto N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector_host : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      len : number of elements in input vector. Cannot be greater than amount reseved during constructor
*/
void U256::addm(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  copyVectorToDevice(in_vector_host, len);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/2 + blockD - 1) / blockD;
  addmu256_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, p, in_vector_len, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, len/2);
}
/*
    Modulo p

    Arguments :
      in_vector_host : Input vector of upto N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector_host : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      len : number of elements in input vector. Cannot be greater than amount reseved during constructor
*/
void U256::mod(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len)
{
  if (len > in_vector_len) { return; }

  copyVectorToDevice(in_vector_host, len);

  // perform sub operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len + blockD - 1) / blockD;
  modu256_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device,p, in_vector_len);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, len);
}
/*
    Modular sub

    Arguments :
      in_vector_host : Input vector of upto N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector_host : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      len : number of elements in input vector. Cannot be greater than amount reseved during constructor
*/
void U256::subm(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  copyVectorToDevice(in_vector_host, len);

  // perform sub operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/2 + blockD - 1) / blockD;
  submu256_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, p, in_vector_len, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, len/2);
}
/*
    Montgomery Multiplication

    Arguments :
      in_vector_host : Input vector of upto N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector_host : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      len : number of elements in input vector. Cannot be greater than amount reseved during constructor
*/
void U256::mulmont(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, uint32_t np, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  copyVectorToDevice(in_vector_host, len);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/2 + blockD - 1) / blockD;
  mulmontu256_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, p, np, in_vector_len, premod);
  //mulmontu256_kernel<<<1,1>>>(out_vector_device, in_vector_device, p, np, in_vector_len, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, len/2);
}

/*
    Transfer input vector from host to device

    Arguments :
      in_vector_host : Input vector of upto N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      len : number of elements in input vector to be xferred. 
          Cannot be greater than amount reseved during constructor, but not checked
*/
void U256::copyVectorToDevice(const uint32_t *in_vector_host, uint32_t len)
{
  uint32_t size = len * sizeof(uint32_t) * NWORDS_256BIT ;
  // Copy input data to device memory
  CCHECK(cudaMemcpy(in_vector_device, in_vector_host, size, cudaMemcpyHostToDevice));
}

/*
    Transfer output vector from device to host

    Arguments :
      out_vector_host : Output vector of upto N/2 256 bit elements Y[0], Y[1], Y[2] ... Y[N/2-1].
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
void U256::copyVectorFromDevice(uint32_t *out_vector_host, uint32_t len)
{
  uint32_t size = len * sizeof(uint32_t) * NWORDS_256BIT ;
  
  // copy results from device to host
  CCHECK(cudaMemcpy(out_vector_host, out_vector_device, size, cudaMemcpyDeviceToHost));
  CCHECK(cudaGetLastError());
}

U256::~U256()
{
  cudaFree(in_vector_device);
  cudaFree(out_vector_device);
  cudaFree(p);
}
