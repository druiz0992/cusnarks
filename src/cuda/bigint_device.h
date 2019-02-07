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
// File name  : bigint_device.h
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of big integer device functionality
// ------------------------------------------------------------------

*/
#ifndef _BIGINT_DEVICE_H_
#define _BIGINT_DEVICE_H_

__global__ void addm_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector);
__global__ void montmul_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector);
__forceinline__ __device__ void umadc32(uint32_t x, uint32_t y, uint32_t a, uint32_t *z, uint32_t *c);
__forceinline__ __device__ void ucprop32(uint32_t *x, uint32_t *c);
__forceinline__ __device__ void uadd256(const uint32_t *x, const uint32_t *y, uint32_t *z);
__forceinline__ __device__ void uaddm256(const uint32_t *x, const uint32_t *y, uint32_t *z, const uint32_t *p);
__forceinline__ __device__ void usub256(const uint32_t *x, const uint32_t *y, uint32_t *z);

#endif
