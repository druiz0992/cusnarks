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
// File name  : cusnarks_kernel.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of Cusnarks CUDA resources management
//   
// ------------------------------------------------------------------

*/

#include <assert.h>
#include <iostream>
#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "rng.h"
#include "cusnarks_kernel.h"


using namespace std;

/*
    Constructor : Reserves device memory for vector and modulo p. 

    Arguments :
      p : 256 bit number in 8 word uint32 array
      length : Vector length for future arithmetic operations
*/
CUSnarks::CUSnarks (const uint32_t *p, uint32_t device_vector_len, uint32_t in_size, uint32_t out_size)
{
  CUSnarks(p, device_vector_len, in_size, out_size 0);
}

CUSnarks::CUSnarks (const uint32_t *p, uint32_t device_vector_len, uint32_t in_size, uint32_t out_size, uint32_t seed) : in_vector_len(device_vector_len)
{
  allocateCudaResources(p, in_size, out_size);
  initRNG(seed);
}

void CUSnarks::allocateCudaResources(const uint32_t *p, uint32_t in_size, uint32_t out_size)
{
  // Allocate global memory in device for input and output
  CCHECK(cudaMalloc((void**) &this->in_vector_device, in_size));

  CCHECK(cudaMalloc((void**) &this->out_vector_device, out_size);

  // Allocate global memory for modulo p
  CCHECK(cudaMalloc((void**) &this->p, sizeof(uint32_t) * NWORDS_256BIT));

  // Copy modulo p to device memory
  CCHECK(cudaMemcpy(this->p, p, sizeof(uint32_t) * NWORDS_256BIT, cudaMemcpyHostToDevice));
}
void CUSnarks::initRNG(uint32_t seed)
{
  if (seed == 0){ rng =  _RNG::get_instance(); }
  else { rng = _RNG::get_instance(seed); }
}

void CUSnarks::rand(uint32_t *samples, uint32_t n_samples, uint32_t size)
{
    rng->randu32(samples, n_samples * size);
}


/*
    Transfer input vector from host to device

    Arguments :
      in_vector_host : Input vector of upto N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      len : number of elements in input vector to be xferred. 
          Cannot be greater than amount reseved during constructor, but not checked
*/
void CUSnarks::copyVectorToDevice(const uint32_t *in_vector_host, uint32_t size)
{
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
void CUSnarks::copyVectorFromDevice(uint32_t *out_vector_host, uint32_t out_size)
{
  // copy results from device to host
  CCHECK(cudaMemcpyHostToDeviceMemcpy(out_vector_host, out_vector_device, out_size, cudaMemcpyDeviceToHost));
  CCHECK(cudaGetLastError());
}

CUSnarks::~CUSnarks()
{
  cudaFree(in_vector_device);
  cudaFree(out_vector_device);
  cudaFree(p);
}

template<typename kernelFunction, typename... kernelParameters>
void CUSnarks::kernelLaunch(
		uint32_t *out_vector_host,
	       	const uint32_t *in_vector_host,
	        uint32_t in_size,
		uint32_t out_size,
		const kernelFunction& kernel_function,
		kernelParameters... kernel_extra_params)
{
  if (len > in_vector_len) { return; }

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/6 + blockD - 1) / blockD;
  kernel_function<<<gridD, blockD>>>(out_vector_device, in_vector_device, p, len, kernel_params...);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());
  copyVectorFromDevice(out_vector_host, out_size);
}
