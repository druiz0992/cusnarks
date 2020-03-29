#!/usr/bin/python3
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
from cuda_wrapper import *

logging.basicConfig(level=logging.DEBUG,
                    format='[%(processName)s] %(asctime)s %(levelname)s %(message)s')


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
from pycusnarks import *

sys.path.append('../../src/python')

def profile_ecbn128():

    #nkernels = 2
    nsamples = 1<<20
    sort_en = 0
    n_gpu = 1
    max_streams = N_STREAMS_PER_GPU - 1
    #n_streams = max_streams
    n_streams = -1
    if n_streams < 0:
        n_streams=0

    niter = n_streams * n_gpu
    if niter == 0:
        niter = n_gpu

    curve_data = ZUtils.CURVE_DATA['BN128']

    prime = ZUtils.CURVE_DATA['BN128']['prime']
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    ECC.init(curve_data['curve_params'])
    pu256 = BigInt(prime).as_uint256()

    midx = MOD_GROUP
    Gx = to_montgomeryN_h(np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32), midx)[0]
    Gy = to_montgomeryN_h(np.asarray([2,0,0,0,0,0,0,0], dtype=np.uint32), midx)[0]
    G = np.concatenate((Gx,Gy,Gx))

    cu_ecbn128 = ECBN128(nsamples+3, seed=560)


    #my_kernels = [CB_EC_JACAFF_ADD , CB_EC_JAC_ADD , CB_EC_JACAFF_DOUBLE,
                  #CB_EC_JAC_DOUBLE, CB_EC_JAC_MUL, CB_EC_JAC_MUL1, CB_EC_JAC_MAD_SHFL]
    #my_kernels = [CB_EC_JAC_MAD_SHFL]
    #my_kernels = [CB_EC_JAC_ADD, CB_EC_JAC_DOUBLE]
    #my_kernels = [CB_EC_JAC_ADD]
    #my_kernels = [CB_EC_JAC_MUL_OPT]
    #my_kernels = [CB_EC_JAC_MAD_SHFL]
    #my_kernels = [CB_EC_JAC_RED]

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
          nkernels = 1
  
      if kernel_config['kernel_idx'][0] == CB_EC_JACAFF_ADD or \
         kernel_config['kernel_idx'][0] == CB_EC_JAC_ADD:
         kernel_params['out_length'] = int(3*nsamples/2)
         kernel_params['stride'] = [2 * 3]
         nkernels = 1
      
      if kernel_config['kernel_idx'][0] == CB_EC_JAC_MUL_OPT and \
              len(kernel_config['kernel_idx']) == 1:

         kernel_params['in_length'] = [3*nsamples]
         kernel_params['out_length'] = int(3*nsamples/U256_BSELM)
         kernel_params['stride'] = [U256_BSELM * 3]
         nkernels = 1

      niter = 1
      for i in range(niter):
         gpu_id=i%n_gpu
         if n_streams:
           stream_id = int(i/n_gpu)+1
         else:
           stream_id = 0

         print(gpu_id, stream_id)

         ecbn128_vector = cu_ecbn128.randu256(nsamples,pu256)
         tmp = ec_jacscmulx1_h(np.reshape(ecbn128_vector[:nsamples*NWORDS_256BIT],-1), G, midx, 0)

         if my_kernels[0] == CB_EC_JAC_MAD_SHFL or \
                 my_kernels[0] == CB_EC_JAC_RED or \
                 my_kernels[0] == CB_EC_JAC_MUL_OPT:
             tmp = ec_jac2aff_h(np.reshape(tmp,-1), midx, 1)
             ecbn128_vector = np.concatenate((ecbn128_vector, tmp))
             idx_v = sortu256_idx_h(ecbn128_vector[:nsamples],sort_en)
             input_vector = np.concatenate((ecbn128_vector[:nsamples][idx_v], ecbn128_vector[nsamples:]))
         else:
             input_vector = tmp


         if kernel_config['kernel_idx'][0] == CB_EC_JAC_MAD_SHFL or \
            kernel_config['kernel_idx'][0] == CB_EC_JAC_RED:
            separate_k = 1

            if kernel_config['kernel_idx'][0] == CB_EC_JAC_MAD_SHFL :
                separate_k = 0

            a, r,kernel_time = ec_mad_cuda2(cu_ecbn128, input_vector, midx, 0, shamir_en=1, gpu_id=0, stream_id=0,separate_k=separate_k)

         else:
           print(kernel_params, kernel_config)
           r,kernel_time = cu_ecbn128.kernelLaunch(input_vector, kernel_config, kernel_params,gpu_id, n_streams,n_kernels=nkernels)
         if i :
             kernel_stats.append(kernel_time)
      
      if niter > 1:
        logging.info("Kernel %s : - Max : %s [s], Min : %s [s], Mean : %s[s]" % (k, np.max(kernel_stats), np.min(kernel_stats), np.mean(kernel_stats)))

     

if __name__ == "__main__":
    profile_ecbn128()
