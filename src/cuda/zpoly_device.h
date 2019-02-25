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


__global__ void fft_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void ifft_kernel_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);

#endif
