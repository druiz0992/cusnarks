"""
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
*/

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : prof_u256.py
//
// Date       : 22/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Ptofile u256 CUDA Module
//
// ------------------------------------------------------------------

"""
import os,sys, os.path
import numpy as np
import logging

sys.path.append('../../src/python')

from bigint import *
from zutils import *
from zfield import *
from constants import *

logging.basicConfig(level=logging.DEBUG,
                    format='[%(processName)s] %(asctime)s %(levelname)s %(message)s')


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
from pycusnarks import *

sys.path.append('../../src/python')
from bigint import *

def profile_u256():

    niter = 10
    kernel_stats = []
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    nsamples = 1<<22

    kernel_params = {'midx' : [MOD_FIELD] ,'premod' : [0], 'in_length' :[nsamples], 'stride' : [2], 'out_length' : nsamples/2}
    kernel_config = {'blockD' : [256], 'smemS' : [0]}
    nkernels = 1

    u256 = U256(nsamples, seed=10)
    u256_vector = u256.rand(nsamples)
            
    kernel_config['kernel_idx'] = [CB_U256_ADDM_REDUCE]

    if kernel_config['kernel_idx'][0] == CB_U256_MOD or kernel_params['premod'][0] == 1:
      for s in u256_vector:
         s |= 0xF0000000
    elif kernel_config['kernel_idx'][0] == CB_U256_ADDM_REDUCE:
        kernel_params['midx'] = [MOD_FIELD, MOD_FIELD] 
        kernel_params['premod'] = [1, 0]
        kernel_params['stride'] = [8, 8]
        kernel_config['blockD'] = [256, 256]
        kernel_params['in_length'] = [nsamples, \
                                         (nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]) -1) / (kernel_config['blockD'][0]*kernel_params['stride'][0])]
        kernel_params['out_length'] = 1
        kernel_config['smemS'] = [kernel_config['blockD'][0] * NWORDS_256BIT * 4, \
                                     kernel_config['blockD'][1] * NWORDS_256BIT * 4]
        kernel_config['gridD'] = [0,  1]
        kernel_config['kernel_idx'] = [CB_U256_ADDM_REDUCE, CB_U256_ADDM_REDUCE]
        nkernels = 2

    for i in range(niter):
       u256 = U256(nsamples, seed=10)
       _, kernel_time = u256.kernelLaunch(u256_vector, kernel_config, kernel_params, nkernels )
       if i :
           kernel_stats.append(kernel_time)

    
    logging.info("Max : %s [s], Min : %s [s], Mean : %s[s]" % (np.max(kernel_stats), np.min(kernel_stats), np.mean(kernel_stats)))

if __name__ == "__main__":
    profile_u256()
