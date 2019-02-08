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
//  Definition of U256 integer device functionality
// ------------------------------------------------------------------

*/
#ifndef _U256_DEVICE_H_
#define _U256_DEVICE_H_

__global__ void addu256_kernel(uint32_t *in_vector, uint32_t len, uint32_t *out_vector);
__global__ void subu256_kernel(uint32_t *in_vector, uint32_t len, uint32_t *out_vector);
__global__ void addmu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector);
__global__ void submu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector);
__global__ void modu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector);
__global__ void mulmontu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector);

__forceinline__ __device__ void addu256(const uint32_t *x, const uint32_t *y, uint32_t *z);
__forceinline__ __device__ void subu256(const uint32_t *x, const uint32_t *y, uint32_t *z);
__forceinline__ __device__ void addmu256(const uint32_t *x, const uint32_t *y, uint32_t *z, const uint32_t *p);
__forceinline__ __device__ void submu256(const uint32_t *x, const uint32_t *y, uint32_t *z, const uint32_t *p);
__forceinline__ __device__ void modu256(const uint32_t *x, uint32_t *z, const uint32_t *p);

__forceinline__ __device__ void madcu32(uint32_t x, uint32_t y, uint32_t a, uint32_t *z, uint32_t *c);
__forceinline__ __device__ void propcu32(uint32_t *x, uint32_t *c);

#endif
