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
// File name  : u256_device.h
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of U256 integer arithmetic
// ------------------------------------------------------------------

*/
#ifndef _U256_DEVICE_H_
#define _U256_DEVICE_H_

#define U256_MAX_SMALLK (32)

__global__ void addmu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void addmu256_reduce_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void submu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void modu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void mulmontu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void mulmontu256_2_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void shr1u256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);

//template <typename T1, typename T2>
//__forceinline__ __device__ void addu256(T1 *z, T2 *x, T2 y);
__forceinline__ __device__ void addu256(uint32_t __restrict__ *z, const uint32_t  __restrict__ *x, const uint32_t __restrict__ *y);
__forceinline__ __device__ void addu288(uint32_t __restrict__ *z, const uint32_t  __restrict__ *x, const uint32_t __restrict__ *y);
__forceinline__ __device__ void addu256_2(volatile uint32_t *z,  const volatile uint32_t *x, const volatile uint32_t *y);

//__forceinline__ __device__ void addu256_2(volatile uint32_t *z,  const volatile uint32_t *x, const volatile uint32_t *y);
__forceinline__ __device__ uint32_t ltu256(const uint32_t __restrict__ *x, const uint32_t __restrict__ *y);
__forceinline__ __device__ void subu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y);
__forceinline__ __device__ uint32_t subgtu256(uint32_t __restrict__ *x, const uint32_t __restrict__ *y);


template <typename T1, typename T2>
__device__ void addmu256_2(T1 *z, T2  *x, T2 *y, mod_t midx);
//__device__ void addmu256_2(volatile uint32_t *z, const volatile uint32_t *x, const volatile uint32_t *y, mod_t midx);
//template <typename T1, typename T2>
//extern __device__ void addmu256(T1 *z, T2 *x, T2 *y, mod_t midx);
extern __device__ void addmu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y, mod_t midx);
extern __device__ void submu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y, mod_t midx);
extern __device__ void modu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, mod_t midx);
extern __device__ void mulmontu256(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, const uint32_t __restrict__ *B, mod_t midx);
extern __device__ void sqmontu256(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, mod_t midx);
extern __device__ void mulku256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ k, mod_t midx);
extern __device__ void mulmontu256_2(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, const uint32_t __restrict__ *B, mod_t midx);
extern __device__ void sqmontu256_2(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, mod_t midx);
extern __device__ uint32_t shr1u256(const uint32_t __restrict__ *x);


#endif
