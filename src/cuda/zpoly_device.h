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
// File name  : zpoly_device.h
//
// Date       : 25/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of ZPoly arithmetic
// ------------------------------------------------------------------

*/
#ifndef _ZPOLY_DEVICE_H_
#define _ZPOLY_DEVICE_H_

#define ZPOLY_FFT_32 (5)

__global__ void zpoly_fft32_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void zpoly_ifft32_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void zpoly_mul32_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);

__global__ void zpoly_fftN_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void zpoly_ifftN_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void zpoly_mulN_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);

__global__ void zpoly_fft2DX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void zpoly_fft2DY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void zpoly_fft3D_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);

__device__ void fft32_dif(uint32_t *z, uint32_t *x, mod_t midx);
__device__ void ifft32_dit(uint32_t *z, uint32_t *x, mod_t midx);
__device__ void fftN_dif(uint32_t *z, uint32_t *x, uint32_t N, mod_t midx);
__device__ void ifftN_dit(uint32_t *z, uint32_t *x, uint32_t N, mod_t midx);

__device__ void mul_poly(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t d, mod_t midx);
__device__ void fft2Dx_dif(uint32_t *z, uint32_t *x, uint32_t Nx, uint32_t Ny, mod_t midx);
__device__ void fft2Dy_dif(uint32_t *z, uint32_t *x, uint32_t Nx, uint32_t Ny, mod_t midx);

__forceinline__ __device__ void fft_butterfly(uint32_t *d_out, uint32_t *d_in, uint32_t srcLane );
#endif
