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
// File name  : ecbn128.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of CUDA EC Kernel processing
//
//
//  General  Kernel input vector format:
//     N groups, where each group is made of 1 256 bit scalar number
//    and one Elliptic Point with two 256 bit coordinates
// 
//     X[0], PX[0], PY/Z[0], X[1], PX[1], PY/Z[1],..., X[N-1], PX[N-1], PY/Z[N-1]
//
//  Kernels
// {addec_kernel, doublec_kernel, scmulec_kernel, addec_reduce_kernel, scmulec_reduce_kernel};
//
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
#include "ecbn128.h"
#include "ecbn128_device.h"

using namespace std;

static kernel_cb ecbn128_kernel_callbacks[] = {addecldr_kernel, doublecldr_kernel, scmulecldr_kernel, addecldr_reduce_kernel, scmulecldr_reduce_kernel};

ECBN128::ECBN128 (uint32_t len) : CUSnarks( len, NWORDS_256BIT * sizeof(uint32_t) * len * (ECPOINT_NDIMS + U256_NDIMS),
		                            len,  NWORDS_256BIT * sizeof(uint32_t) * len * ECPOINT_NDIMS, 
                                            ecbn128_kernel_callbacks, 0)
{
}

ECBN128::ECBN128 (uint32_t len, const uint32_t seed) :  CUSnarks(len, NWORDS_256BIT * sizeof(uint32_t) * len * (ECPOINT_NDIMS + U256_NDIMS),
				                                 len, NWORDS_256BIT * sizeof(uint32_t) * len * ECPOINT_NDIMS, 
						       ecbn128_kernel_callbacks, seed)
{
}

#if 0
/*
*/
void ECBN128::add(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, mod_t mod_idx, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + U256_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * len/2 * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/6 + blockD - 1) / blockD;
  addec_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, in_vector_len, mod_idx, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}

void ECBN128::doubl(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, mod_t mod_idx, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + U256_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * len * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/3 + blockD - 1) / blockD;
  doublec_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, in_vector_len, mod_idx, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}

void ECBN128::mul(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, mod_t mod_idx, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + U256_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * len * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/3 + blockD - 1) / blockD;
  scmulec_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device,in_vector_len, mod_idx, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}

/*
*/
void ECBN128::add_reduce(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, mod_t mod_idx, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + U256_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/6 + blockD - 1) / blockD;
  addec_reduce_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, in_vector_len, mod_idx,  premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}

void ECBN128::mul_reduce(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, mod_t mod_idx,  uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + U256_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/3 + blockD - 1) / blockD;
  scmulec_reduce_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, in_vector_len, mod_idx, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}
#endif
