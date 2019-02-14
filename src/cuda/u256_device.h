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

__global__ void addmu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void submu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void modu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void mulmontu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);

__forceinline__ __device__ void addu256(uint32_t __restrict__ *z, const uint32_t  __restrict__ *x, const uint32_t __restrict__ *y);
__forceinline__ __device__ void subu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y);
extern __device__ void addmu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y, mod_t midx);
extern __device__ void submu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y, mod_t midx);
extern __device__ void modu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, mod_t midx);
extern __device__ void mulmontu256(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, const uint32_t __restrict__ *B, mod_t midx);
__forceinline__ __device__ uint32_t ltu256(const uint32_t __restrict__ *x, const uint32_t __restrict__ *y);
__forceinline__ __device__ uint32_t eq0u256(const uint32_t __restrict__ *x);
__forceinline__ __device__ void shr1u256(const uint32_t __restrict__ *x);

__forceinline__ __device__ void mulu32(uint32_t __restrict__ *z,  uint32_t x,  uint32_t y);
__forceinline__ __device__ void madcu32(uint32_t *c, uint32_t *s, uint32_t x, uint32_t y, uint32_t a);
__forceinline__ __device__ void addcu32(uint32_t *c, uint32_t *s, uint32_t x, uint32_t y);
__forceinline__ __device__ void propcu32(uint32_t *x, uint32_t c, uint32_t digit);

__device__ void printNumber(uint32_t *n);

#endif
