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
// File name  : ecbn128_device.h
//
// Date       : 12/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of EC Cuda functionality
// ------------------------------------------------------------------

*/
#ifndef _ECBN128_DEVICE_H_
#define _ECBN128_DEVICE_H_

__global__ void addecjac_kernel(uint32_t   *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void doublecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void scmulecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void madecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void madecjac_shfl_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);


template<typename T1, typename T2>
__forceinline__ __device__ void madecjac(T1 *xr, T1 *xo, uint32_t *scl, T1 *smem_ptr, kernel_params_t *params);
template<typename T1, typename T2>
__forceinline__ __device__ void madecjac_shfl(T1 *xr, T1 *xo, uint32_t *scl, T1 *smem_ptr, kernel_params_t *params);
template<typename T1, typename T2>
//__forceinline__ __device__ void addecjac(T1 *xr, T1 *x1, T1 *x2, mod_t midx);
__forceinline__ __device__ void addecjac(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t x1offset, T1 *zx2, uint32_t x2offset, mod_t midx);
template<typename T1, typename T2>
__forceinline__ __device__ void doublecjac(T1 *xr, T1 *x1, mod_t midx);
template <typename T1, typename T2>
__forceinline__ __device__ void addecjacaff(T1  *zxr, T1 *zx1, T1 *zx2, mod_t midx);
template<typename T1, typename T2>
__forceinline__ __device__ void doublecjacaff(T1 *zxr,  T1 *zx1, mod_t midx);
template<typename T1, typename T2>
__forceinline__ __device__ void scmulecjac(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t xoffset, uint32_t *scl, mod_t midx);
template<typename T1, typename T2>
__forceinline__ __device__ void shflxoruecc(T1 *d_out,T1 *d_in, uint32_t srcLane );



#if 0
__global__ void addecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void doublecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void scmulecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void madecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);

__forceinline__ __device__ 
void addecldr(uint32_t *xr, const uint32_t *x1, const uint32_t *x2, const uint32_t *xp, mod_t mod_idx);
__forceinline__ __device__ 
void doublecldr(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, mod_t midx);
__forceinline__ __device__ 
void ldrstep(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, uint32_t *scl, mod_t midx);
#endif


#endif
