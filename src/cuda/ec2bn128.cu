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
// File name  : ec2bn128.cu
//
// Date       : 22/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of CUDA EC Kernel processing for extended fields
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
#include "ec2bn128.h"
#include "ecbn128_device.h"

using namespace std;

static kernel_cb ecbn128_2_kernel_callbacks[] = {addec2jacaff_kernel, addec2jac_kernel, doublec2jacaff_kernel, doublec2jac_kernel,
                                                 scmulec2jac_kernel, sc1mulec2jac_kernel, madec2jac_kernel, madec2jac_shfl_kernel};

EC2BN128::EC2BN128 (uint32_t len) : CUSnarks( len * ECP2_JAC_INDIMS, NWORDS_256BIT * sizeof(uint32_t) * len *  ECP2_JAC_INDIMS,
		                            len * ECP2_JAC_OUTDIMS,  NWORDS_256BIT * sizeof(uint32_t) * len * ECP2_JAC_OUTDIMS, 
                                            ecbn128_2_kernel_callbacks, 0)
{
}

EC2BN128::EC2BN128 (uint32_t len, const uint32_t seed) :  CUSnarks(len * ECP2_JAC_INDIMS, NWORDS_256BIT * sizeof(uint32_t) * len * ECP2_JAC_INDIMS,
				                                 len * ECP2_JAC_OUTDIMS, NWORDS_256BIT * sizeof(uint32_t) * len * ECP2_JAC_OUTDIMS,
						       ecbn128_2_kernel_callbacks, seed)
{
}

