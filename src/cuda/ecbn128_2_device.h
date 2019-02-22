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
// File name  : ecbn128_2_device.h
//
// Date       : 12/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of EC Cuda functionality extended fields
// ------------------------------------------------------------------

*/
#ifndef _ECBN128_2_DEVICE_H_
#define _ECBN128_2_DEVICE_H_

__global__ void addecjac2_kernel(uint32_t   *out_vector_data, uint32_t *in_vector_data, kernel_params_t *params);
__global__ void doublecjac2_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void scmulecjac2_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void madecjac2_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);

__forceinline__ __device__
void addecjac2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, const uint32_t *x2, mod_t midx);
__forceinline__ __device__
void doublecjac2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, mod_t midx);
__forceinline__ __device__
void addecjacaff2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, const uint32_t *x2, mod_t midx);
__forceinline__ __device__
void doublecjacaff2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, mod_t midx);

__forceinline__ __device__
void scmulecjac2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, uint32_t *scl, mod_t midx);
#endif
