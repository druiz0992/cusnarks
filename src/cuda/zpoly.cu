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
// File name  : zpoly.cu
//
// Date       : 25/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of >Poly kernel processing
// 
//  General  Kernel input vector format:
//     NxM 256 bit coefficients X[0], X[1],....,X[N-1] for M polynomials of degree N-1
//
// TODO
// ------------------------------------------------------------------

*/

#include <assert.h>
#include <iostream>
#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "rng.h"
#include "cusnarks_kernel.h"
#include "zpoly.h"
#include "zpoly_device.h"


using namespace std;

// kernel callbacks indexed by XXX_callback_t enum type
static kernel_cb zpoly_kernel_callbacks[] = {zpoly_fft32_kernel,
                                             zpoly_ifft32_kernel, 
                                             zpoly_mul32_kernel,
                                             zpoly_fftN_kernel,  
                                             zpoly_ifftN_kernel,
                                             zpoly_mulN_kernel, 
                                             zpoly_fft2DX_kernel,
                                             zpoly_fft2DY_kernel,
                                             zpoly_fft3DXX_kernel, 
                                             zpoly_fft3DXXprev_kernel,
                                             zpoly_fft3DXY_kernel,
                                             zpoly_fft3DYX_kernel,
                                             zpoly_fft3DYY_kernel,
                                             zpoly_interp3DXX_kernel, 
                                             zpoly_interp3DXY_kernel,
                                             zpoly_interp3DYX_kernel,
                                             zpoly_interp3DYY_kernel,
                                             zpoly_interp3Dfinish_kernel,
                                             zpoly_fft4DXX_kernel,
                                             zpoly_fft4DXY_kernel,
                                             zpoly_fft4DYX_kernel,
                                             zpoly_fft4DYY_kernel,
                                             zpoly_interp4DXX_kernel, 
                                             zpoly_interp4DXY_kernel,
                                             zpoly_interp4DYX_kernel,
                                             zpoly_interp4DYY_kernel,
                                             zpoly_interp4Dfinish_kernel,
                                             zpoly_add_kernel,
                                             zpoly_sub_kernel,
                                             zpoly_subprev_kernel, 
                                             zpoly_mulc_kernel,
                                             zpoly_mulcprev_kernel,
                                             zpoly_mulK_kernel, 
                                             zpoly_madprev_kernel,
                                             zpoly_addprev_kernel,
                                             zpoly_divsnarks_kernel};

/*
    Constructor : Reserves GPU memory and data initialization.


    Arguments :
*/
ZCUPoly::ZCUPoly (uint32_t len) : CUSnarks( len, NWORDS_256BIT * sizeof(uint32_t) * len,
                                      len, NWORDS_256BIT * sizeof(uint32_t) * len,
				      zpoly_kernel_callbacks, 0)
{
}

ZCUPoly::ZCUPoly (uint32_t len, uint32_t seed) : CUSnarks(len, NWORDS_256BIT * sizeof(uint32_t) * len,
	                                            len, NWORDS_256BIT * sizeof(uint32_t) * len, 
				                    zpoly_kernel_callbacks, seed)
{
}

