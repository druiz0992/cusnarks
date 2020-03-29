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
// File name  : prof_ec2bn128.py
//
// Date       : 05/08/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Profile EC2BN128 CUDA Module
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

def profile_ec2bn128():

    nkernels = 1
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
    G = to_montgomeryN_h( np.asarray([45883430, 2390996433, 1232798066, 3706394933, 2541820639, 4223149639, 2945863739,  425146433,
                          2288773622, 1637743261, 4120812408, 4269789847,  589004286, 4288551522, 2929607174,  687701739,   
                          2823577920, 2947838845, 1476581572, 1615060314, 1386229638,  166285564,  988445547,  352252035,  
                          3340261102, 1678334806,  847068347, 3696752930,  859115638, 1442395582, 2482857090,  228892902,
                          1,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0], dtype=np.uint32),midx)
    G = np.reshape(G,-1)

    cu_ec2bn128 = EC2BN128(2*nsamples+3, seed=560)

    #my_kernels = [CB_EC2_JACAFF_ADD , CB_EC2_JAC_ADD , CB_EC2_JACAFF_DOUBLE,
                  #CB_EC2_JAC_DOUBLE, CB_EC2_JAC_MUL, CM_EC2_JAC_MUL1, CB_EC2_JAC_MAD_SHFL]
    #my_kernels = [CB_EC2_JAC_ADD]
    #my_kernels = [CB_EC2_JAC_ADD, CB_EC2_JAC_DOUBLE]
    #my_kernels = [CB_EC2_JAC_DOUBLE]
    #my_kernels = [CB_EC2_JAC_MUL_OPT]
    #my_kernels = [CB_EC2_JAC_MAD_SHFL]
    my_kernels = [CB_EC2_JAC_RED]

    for k in my_kernels:
      kernel_params = {}
      kernel_config = {}
      kernel_stats = []

      # Test ECBN kernel:
      kernel_params['in_length'] = [6*nsamples]
      kernel_params['out_length'] = 6*nsamples
      kernel_params['stride'] = [3]
      kernel_params['premod'] = [0]
      kernel_params['midx'] = [MOD_GROUP]
  
      kernel_config['smemS'] = [0]
      kernel_config['blockD'] = [128]
      kernel_config['gridD'] = [0]
      kernel_config['kernel_idx'] = [k]
     
      if kernel_config['kernel_idx'][0] == CB_EC2_JACAFF_ADD:
            kernel_params['in_length'] = [nsamples * ECP2_JAC_INDIMS]
            kernel_params['out_length'] = int((nsamples * ECP2_JAC_OUTDIMS) / 2)
            kernel_params['stride'] = [2 * ECP2_JAC_INDIMS]

      if kernel_config['kernel_idx'][0] == CB_EC2_JAC_ADD :
            kernel_params['in_length'] = [nsamples * ECP2_JAC_OUTDIMS]
            kernel_params['out_length'] = (nsamples * ECP2_JAC_OUTDIMS) / 2
            kernel_params['stride'] = [2 * ECP2_JAC_OUTDIMS]


      if kernel_config['kernel_idx'][0] == CB_EC2_JACAFF_DOUBLE :
            kernel_params['in_length'] = [nsamples * ECP2_JAC_INDIMS]
            kernel_params['out_length'] = (nsamples * ECP2_JAC_OUTDIMS)
            kernel_params['stride'] = [1 * ECP2_JAC_INDIMS]

      if kernel_config['kernel_idx'][0] == CB_EC2_JAC_DOUBLE :
            kernel_params['in_length'] = [nsamples * ECP2_JAC_OUTDIMS]
            kernel_params['out_length'] = (nsamples * ECP2_JAC_OUTDIMS)
            kernel_params['stride'] = [1 * ECP2_JAC_OUTDIMS]

      if kernel_config['kernel_idx'][0] == CB_EC2_JAC_MUL :
            kernel_params['in_length'] = [nsamples * (ECP2_JAC_INDIMS + U256_NDIMS)]
            kernel_params['out_length'] = (nsamples * ECP2_JAC_OUTDIMS)
            kernel_params['stride'] = [1 * (ECP2_JAC_INDIMS + U256_NDIMS)]

      if kernel_config['kernel_idx'][0] == CB_EC2_JAC_MUL_OPT and \
              len(kernel_config['kernel_idx']) == 1:

         kernel_params['in_length'] = [5*nsamples]
         kernel_params['out_length'] = int(6*nsamples/U256_BSELM)
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

         ec2bn128_vector = cu_ec2bn128.randu256(nsamples,pu256)
         tmp = ec2_jacscmulx1_h(np.reshape(ec2bn128_vector[:nsamples*NWORDS_256BIT],-1), G, midx, 0)

         if my_kernels[0] == CB_EC2_JAC_MAD_SHFL or \
                 my_kernels[0] == CB_EC2_JAC_RED or \
                 my_kernels[0] == CB_EC2_JAC_MUL_OPT:
           tmp = ec2_jac2aff_h(np.reshape(tmp,-1), midx, 1)
           ec2bn128_vector = np.concatenate((ec2bn128_vector, tmp))
           idx_v = sortu256_idx_h(ec2bn128_vector[:nsamples],sort_en)
           input_vector = np.concatenate((ec2bn128_vector[:nsamples][idx_v], ec2bn128_vector[nsamples:]))
         else:
             input_vector = tmp

         if kernel_config['kernel_idx'][0] == CB_EC2_JAC_MAD_SHFL or \
            kernel_config['kernel_idx'][0] == CB_EC2_JAC_RED:
            separate_k = 1

            if kernel_config['kernel_idx'][0] == CB_EC2_JAC_MAD_SHFL :
                separate_k = 0

            a, r,kernel_time = ec_mad_cuda2(cu_ec2bn128, input_vector, midx, 1, shamir_en=1, gpu_id=0, stream_id=0,separate_k=separate_k)

         else:
           print(kernel_params, kernel_config)
           r,kernel_time = cu_ec2bn128.kernelLaunch(input_vector, kernel_config, kernel_params,gpu_id,n_streams,n_kernels = nkernels)
         if i :
             kernel_stats.append(kernel_time)
  
      
      if niter > 1:
        logging.info("Kernel %s : - Max : %s [s], Min : %s [s], Mean : %s[s]" % (k, np.max(kernel_stats), np.min(kernel_stats), np.mean(kernel_stats)))
  
  
if __name__ == "__main__":
    profile_ec2bn128()




