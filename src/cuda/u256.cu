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
// File name  : u256.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of 256 bit unsigned integer kernel processing for the
// following kernels
// 
//  addmu256_kernel -> modular addition
//
//    Kernel input vector format:
//     N 256 bit numbers X[0], X[1],....,X[N-1]
//    Kernel output vector format:
//     N/2 256 bit numbers X[0]+X[1], X[2]+X[3],..., X[N-2] + X[N-1]
//    output length = 2 * input legth 
//    stride = 2
//    params : premod/input_len/mod idx
//
//  submu256_kernel -> modular substraction
//
//    Kernel input vector format:
//     N 256 bit numbers X[0], X[1],....,X[N-1]
//    Kernel output vector format:
//     N/2 256 bit numbers X[0]-X[1], X[2]-X[3],..., X[N-2] - X[N-1]
//    output length = 2 * input legth 
//    stride = 2
//    params : premod/input_len/mod idx
//
//  modu256_kernel -> modulo operation
//
//    Kernel input vector format:
//     N 256 bit numbers X[0], X[1],....,X[N-1]
//    Kernel output vector format:
//     N 256 bit numbers X[0], +X[1],...X[N-1]
//    output length = input legth 
//    stride = 1
//    params : input_len/mod idx
//
//  mulmontu256_kernel -> montgomery multiplication
//
//    Kernel input vector format:
//     N 256 bit numbers X[0], X[1],....,X[N-1] in Montgomery format
//    Kernel output vector format:
//     N/2 256 bit numbers X[0]*X[1], X[2]*X[3],..., X[N-2]*X[N-1]
//           in Montgomery format
//    output length = 2 * input legth 
//    stride = 2
//    params : premod/input_len/mod idx
//
//   
// TODO
//    - Use managed memory for data
//    - Convert from Montgomery to normal represenation
// ------------------------------------------------------------------

*/

#include <assert.h>
#include <iostream>
#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "rng.h"
#include "cusnarks_kernel.h"
#include "u256.h"
#include "u256_device.h"


using namespace std;

// kernel callbacks indexed by XXX_callback_t enum type
static kernel_cb u256_kernel_callbacks[] = {addmu256_kernel, submu256_kernel, modu256_kernel, mulmontu256_kernel};

/*
    Constructor : Reserves GPU memory and data initialization.


    Arguments :
*/
U256::U256 (uint32_t len) : CUSnarks( len, NWORDS_256BIT * sizeof(uint32_t) * len,
                                      len, NWORDS_256BIT * sizeof(uint32_t) * len,
				      u256_kernel_callbacks, 0)
{
}

U256::U256 (uint32_t len, uint32_t seed) : CUSnarks(len, NWORDS_256BIT * sizeof(uint32_t) * len,
	                                            len, NWORDS_256BIT * sizeof(uint32_t) * len, 
				                    u256_kernel_callbacks, seed)
{
}

