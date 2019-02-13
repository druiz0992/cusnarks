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
//  Implementation of CUDA EC BN128 arithmetic
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
#include "ecbn128.h"
#include "ecbn128_device.h"

using namespace std;

ECBN128::ECBN128 (const uint32_t *p, uint32_t device_vector_len) : CUSnarks(p, device_vector_len,
                                                       NWORDS_256BIT * sizeof(uint32_t) * device_vector_len * (ECPOINT_NDIMS + SCALAR_NDIMS),
						       NWORDS_256BIT * sizeof(uint32_t) * device_vector_len * ECPOINT_NDIMS, 
						       0)
{
}

ECBN128::ECBN128 (const uint32_t *p, uint32_t device_vector_len, const uint32_t seed) :  CUSnarks(p, device_vector_len,
                                                       NWORDS_256BIT * sizeof(uint32_t) * device_vector_len * (ECPOINT_NDIMS + SCALAR_NDIMS),
						       NWORDS_256BIT * sizeof(uint32_t) * device_vector_len * ECPOINT_NDIMS, 
						       seed)
{
  // Scalar, EC_X[0], EC_Z[0]??
}

/*
*/
void ECBN128::add(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + SCALAR_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * len/2 * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/6 + blockD - 1) / blockD;
  addec_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, p, in_vector_len, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}

void ECBN128::doubl(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + SCALAR_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * len * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/3 + blockD - 1) / blockD;
  doublecc_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, p, in_vector_len, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}

void ECBN128::mul(uint32_t *out_vector_host, const uint32_t *in_vector_host, const uint32_t *in_scalar_vector_host, uint32_t len, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + SCALAR_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * len * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/3 + blockD - 1) / blockD;
  scmulecc_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, p, in_vector_len, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}

/*
*/
void ECBN128::add_reduce(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + SCALAR_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/6 + blockD - 1) / blockD;
  addec_reduce_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, p, in_vector_len, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}

void ECBN128::mul_reduce(uint32_t *out_vector_host, const uint32_t *in_vector_host, const uint32_t *in_scalar_vector_host, uint32_t len, uint32_t premod)
{
  if (len > in_vector_len) { return; }

  uint32_t in_size = sizeof(uint32_t) * len * NWORDS_256BIT * (ECPOINT_NDIMS + SCALAR_NDIMS);
  uint32_t out_size = sizeof(uint32_t) * NWORDS_256BIT * ECPOINT_NDIMS;

  copyVectorToDevice(in_vector_host, in_size);

  // perform addition operation and leave results in device memory
  int blockD, gridD;
  blockD = U256_BLOCK_DIM;
  gridD = (len/3 + blockD - 1) / blockD;
  scmulecc_reduce_kernel<<<gridD, blockD>>>(out_vector_device, in_vector_device, p, in_vector_len, premod);
  CCHECK(cudaGetLastError());

  CCHECK(cudaDeviceSynchronize());

  copyVectorFromDevice(out_vector_host, out_size);
}

