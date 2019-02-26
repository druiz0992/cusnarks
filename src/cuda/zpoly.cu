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
static kernel_cb zpoly_kernel_callbacks[] = {fft32_kernel, ifft32_kernel};

/*
    Constructor : Reserves GPU memory and data initialization.


    Arguments :
*/
ZPoly::ZPoly (uint32_t len) : CUSnarks( len, NWORDS_256BIT * sizeof(uint32_t) * len,
                                      len, NWORDS_256BIT * sizeof(uint32_t) * len,
				      zpoly_kernel_callbacks, 0)
{
}

ZPoly::Zpoly (uint32_t len, uint32_t seed) : CUSnarks(len, NWORDS_256BIT * sizeof(uint32_t) * len,
	                                            len, NWORDS_256BIT * sizeof(uint32_t) * len, 
				                    zpoly_kernel_callbacks, seed)
{
}

