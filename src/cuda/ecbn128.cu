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
// File name  : ecbn128.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of CUDA EC Kernel processing
//
//
//  General  Kernel input vector format:
//     N groups, where each group is made of 1 256 bit scalar number
//    and one Elliptic Point with two 256 bit coordinates
// 
//     X[0], PX[0], PY/Z[0], X[1], PX[1], PY/Z[1],..., X[N-1], PX[N-1], PY/Z[N-1]
//
//  Kernels
// {addec_kernel, doublec_kernel, scmulec_kernel, addec_reduce_kernel, scmulec_reduce_kernel};
//
//   
// ------------------------------------------------------------------

*/

#include <assert.h>
#include <iostream>
#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "rng.h"
#include "cusnarks_kernel.h"
#include "ecbn128.h"
#include "ecbn128_device.h"

using namespace std;

static kernel_cb ecbn128_kernel_callbacks[] = //{addecldr_kernel, doublecldr_kernel, scmulecldr_kernel, madecldr_kernel,
                                               {addecjacaff_kernel, addecjac_kernel, doublecjacaff_kernel, doublecjac_kernel, 
                                                scmulecjac_kernel, sc1mulecjac_kernel, madecjac_kernel,  madecjac_shfl_kernel,
					       scmulecjacopt_kernel, redecjac_kernel};

ECBN128::ECBN128 (uint32_t len) : CUSnarks( len * (ECP_JAC_INDIMS+U256_NDIMS), NWORDS_256BIT * sizeof(uint32_t) * len *  (ECP_JAC_INDIMS+U256_NDIMS),
		                            len * ECP_JAC_OUTDIMS,  NWORDS_256BIT * sizeof(uint32_t) * len * ECP_JAC_OUTDIMS, 
                                            ecbn128_kernel_callbacks, 0)
{
}

ECBN128::ECBN128 (uint32_t len, const uint32_t seed) :  CUSnarks(len * (ECP_JAC_INDIMS+U256_NDIMS), NWORDS_256BIT * sizeof(uint32_t) * len * (ECP_JAC_INDIMS+U256_NDIMS),
				                                 len * ECP_JAC_OUTDIMS, NWORDS_256BIT * sizeof(uint32_t) * len * ECP_JAC_OUTDIMS,
						       ecbn128_kernel_callbacks, seed)
{
}

