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
// File name  : prof_ecbn128.py
//
// Date       : 03/04/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Profile ECBN128 CUDA Module
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
from zpoly import *
from constants import *
from ecc import *

logging.basicConfig(level=logging.DEBUG,
                    format='[%(processName)s] %(asctime)s %(levelname)s %(message)s')


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
from pycusnarks import *

sys.path.append('../../src/python')
ZPOLY_datafile = '../../data/zpoly_data_1M.npz'

def profile_ecbn128():

    niter = 2
    curve_data = ZUtils.CURVE_DATA['BN128']
    prime = ZUtils.CURVE_DATA['BN128']['prime']
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    ECC.init(curve_data['curve_params'])
    nsamples = 1<<20
    cu_ecbn128 = ECBN128(nsamples+3, seed=560)
    pu256 = BigInt(prime).as_uint256()


    nkernels = 1

    #my_kernels = [CB_EC_JACAFF_ADD , CB_EC_JAC_ADD , CB_EC_JACAFF_DOUBLE,
                  #CB_EC_JAC_DOUBLE, CB_EC_JAC_MUL, CB_EC_JAC_MUL1, CB_EC_JAC_MAD_SHFL]
    #my_kernels = [CB_EC_JAC_MAD_SHFL]
    #my_kernels = [CB_EC_JAC_ADD, CB_EC_JAC_DOUBLE]
    my_kernels = [CB_EC_JAC_MAD_SHFL]

    for k in my_kernels:
      kernel_params = {}
      kernel_config = {}
      kernel_stats = []

      # Test ECBN kernel:
      kernel_params['in_length'] = [3*nsamples]
      kernel_params['out_length'] = 3*nsamples
      kernel_params['stride'] = [3]
      kernel_params['premod'] = [0]
      kernel_params['midx'] = [MOD_GROUP]
  
      kernel_config['smemS'] = [0]
      kernel_config['blockD'] = [256]
      kernel_config['gridD'] = [0]
      kernel_config['kernel_idx']= [k]
  
      if kernel_config['kernel_idx'][0] == CB_EC_JACAFF_ADD or \
         kernel_config['kernel_idx'][0] == CB_EC_JACAFF_DOUBLE:
          kernel_params['in_length'] = [2*nsamples]
          kernel_params['stride'] = [2]
  
      if kernel_config['kernel_idx'][0] == CB_EC_JACAFF_ADD or \
         kernel_config['kernel_idx'][0] == CB_EC_JAC_ADD:
         kernel_params['out_length'] = 3*nsamples/2
         kernel_params['stride'] = [3 * 2]
  
      if kernel_config['kernel_idx'][0] == CB_EC_JAC_MAD_SHFL:
         kernel_params['stride'] = [ECP_JAC_OUTDIMS, ECP_JAC_OUTDIMS, ECP_JAC_OUTDIMS]
         kernel_config['blockD'] = [256,128,32]
         kernel_params['premul'] = [1,0,0]
         kernel_params['premod'] = [0,0,0]
         kernel_params['midx'] = [MOD_GROUP, MOD_GROUP, MOD_GROUP]
         kernel_config['smemS'] = [kernel_config['blockD'][0]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4, \
                                   kernel_config['blockD'][1]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4, \
                                   kernel_config['blockD'][2]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4]
         kernel_config['kernel_idx'] = [CB_EC_JAC_MAD_SHFL, CB_EC_JAC_MAD_SHFL, CB_EC_JAC_MAD_SHFL]
         out_len1 = ECP_JAC_OUTDIMS * ((nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS) -1) /
                                   (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS))
         out_len2 = ECP_JAC_OUTDIMS * ((out_len1 + (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP_JAC_OUTDIMS) -1) /
                                   (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP_JAC_OUTDIMS))
         kernel_params['in_length'] = [nsamples * (ECP_JAC_INDIMS+U256_NDIMS), out_len1, out_len2]
         kernel_params['out_length'] = 1 * ECP_JAC_OUTDIMS
         kernel_params['padding_idx'] = [0,0,0]
         kernel_config['gridD'] = [0,1,1]
         min_length = [ECP_JAC_OUTDIMS * \
             (kernel_config['blockD'][idx] * kernel_params['stride'][idx]/ECP_JAC_OUTDIMS) for idx in range(len(kernel_params['stride']))]
         nkernels = 3
  
      for i in range(niter):
         ecbn128_vector = cu_ecbn128.randu256(3*nsamples,pu256)
         idx_v = sortu256_idx_h(ecbn128_vector[:nsamples])
         input_vector = np.concatenate((ecbn128_vector[:nsamples][idx_v], ecbn128_vector[nsamples:]))
         _,kernel_time = cu_ecbn128.kernelLaunch(input_vector, kernel_config, kernel_params,nkernels)
         if i :
             kernel_stats.append(kernel_time)
  
      
      logging.info("Kernel %s : - Max : %s [s], Min : %s [s], Mean : %s[s]" % (k, np.max(kernel_stats), np.min(kernel_stats), np.mean(kernel_stats)))
  
  
  
if __name__ == "__main__":
    profile_ecbn128()
