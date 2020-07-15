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
import time

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

    init_h()
    #nsamples1 = 1 << 17
    #nsamples =  128*7
    nsamples =  1 << 20

    pippen_binsize = 8
    pippen_blocksize = 8
    pippen_blocks = 4
    max_samples = max(nsamples, 1<<(pippen_blocksize + pippen_binsize + pippen_blocks))
    opt2_blocksize = 8
    #max_samples = nsamples
    
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

    midx = MOD_FP
    #G = np.asarray([45883430, 2390996433, 1232798066, 3706394933, 2541820639, 4223149639, 2945863739,  425146433,
    #                      2288773622, 1637743261, 4120812408, 4269789847,  589004286, 4288551522, 2929607174,  687701739,   
    #                      2823577920, 2947838845, 1476581572, 1615060314, 1386229638,  166285564,  988445547,  352252035,  
    #                      3340261102, 1678334806,  847068347, 3696752930,  859115638, 1442395582, 2482857090,  228892902,
    #                      1,0,0,0,0,0,0,0,
    #                      0,0,0,0,0,0,0,0], dtype=np.uint32)
    #G = np.reshape(G,-1)
    #y2 = montsquare_ext_h(G[16:32], MOD_FP)
    #x2 = montsquare_ext_h(G[0:16], MOD_FP)
    #x3 = montmultN_ext_h(np.reshape(x2,-1), G[0:16], MOD_FP)
    #b = subm_ext_h(np.reshape(y2,-1), np.reshape(x3,-1), MOD_FP)

    cu_ec2bn128 = ECBN128(2*(max_samples+3), seed=560)

    #my_kernels = [CB_EC2_JACAFF_ADD , CB_EC2_JAC_ADD , CB_EC2_JACAFF_DOUBLE,
                  #CB_EC2_JAC_DOUBLE, CB_EC2_JAC_MUL, CM_EC2_JAC_MUL1, CB_EC2_JAC_MAD_SHFL]
    #my_kernels = [CB_EC2_JAC_ADD]
    #my_kernels = [CB_EC2_JAC_ADD, CB_EC2_JAC_DOUBLE]
    #my_kernels = [CB_EC2_JAC_DOUBLE]
    #my_kernels = [CB_EC2_JAC_MUL_OPT]
    #my_kernels = [CB_EC2_JAC_MAD_SHFL]
    #my_kernels = [CB_EC2_JAC_RED]
    my_kernels = [CB_EC2_JAC_MUL_PIPPEN]

    for k in my_kernels:
      kernel_params = {}
      kernel_config = {}
      kernel_stats = []

      # Test ECBN kernel:
      nkernels = 1
      kernel_params['in_length'] = [6*nsamples]
      kernel_params['out_length'] = 6*nsamples
      kernel_params['stride'] = [3]
      kernel_params['premod'] = [0]
      kernel_params['midx'] = [MOD_FP]
  
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

      kernel_config['smemS'] = [0]
      kernel_config['blockD'] = [256]
      kernel_config['gridD'] = [0]
      kernel_config['kernel_idx']= [k]

      if kernel_config['kernel_idx'][0] == CB_EC2_JAC_MUL_PIPPEN :

         nkernels = 4
         nblocks = 1<< pippen_blocks
         nbins =int((NWORDS_FR * NBITS_WORD) / pippen_binsize)
         npoints_out = int((1<<pippen_blocksize) / nbins)
         kernel_params['in_length'] = [5*nsamples,6*nblocks*(1<<(pippen_binsize+pippen_blocksize)), 6*nblocks*(1<<pippen_blocksize), 6*(1<<pippen_blocksize)]
         kernel_params['out_length'] = 6
         #kernel_params['out_length'] = kernel_params['in_length'][1]
         kernel_params['stride'] = [5*int((nsamples+npoints_out*nblocks-1)/(npoints_out*nblocks)),6*(1<<pippen_binsize), 6*nblocks, 6]
         kernel_params['premul'] = [1,0,0,0]
         kernel_params['premod'] = [nblocks,nblocks,nblocks,1]
         kernel_params['padding_idx'] = [pippen_binsize] * nkernels
         kernel_params['midx'] = [MOD_FP] * nkernels

         kernel_config['kernel_idx'] = [CB_EC2_JAC_MUL_PIPPEN, CB_EC2_JAC_RED1_PIPPEN, CB_EC2_JAC_RED2_PIPPEN, CB_EC2_JAC_RED3_PIPPEN]
         kernel_config['blockD'] = [1<<pippen_blocksize] * nkernels
         kernel_config['smemS'] = [0] * nkernels
         kernel_config['smemS'][nkernels-1]   = (1<<(pippen_blocksize-5)) * NWORDS_FP * ECP2_JAC_OUTDIMS * 4
         kernel_config['gridD'] = [nblocks, nblocks, 1, 1]


      niter = 4

      print("Computing random scalars")
      ecbn128_vector = cu_ec2bn128.randuBI(nsamples, 8, pu256)
      #ecbn128_vector = np.reshape(np.asarray([1,0,0,0,0,0,0,0]*nsamples,dtype=np.uint32),(-1,8))
      print("Computing random ECP")
      tmp = ec2_jacsc1mul_h(np.concatenate((np.reshape(ecbn128_vector[:nsamples*NWORDS_256BIT],-1), G)), midx, 0)
      tmp = ec2_jac2aff_h(np.reshape(tmp,-1), midx, 1)
      ecbn128_vector = np.concatenate((ecbn128_vector, tmp))
      idx_v = sortuBI_idx_h(ecbn128_vector[:nsamples],8,sort_en)
      input_vector = np.concatenate((ecbn128_vector[:nsamples][idx_v], ecbn128_vector[nsamples:]))

      for i in range(niter):
         gpu_id=i%n_gpu
         if n_streams:
           stream_id = int(i/n_gpu)+1
         else:
           stream_id = 0

         print(gpu_id, stream_id)

         if my_kernels[0] == CB_EC2_JAC_MAD_SHFL or \
                 my_kernels[0] == CB_EC2_JAC_RED or \
                 my_kernels[0] == CB_EC2_JAC_MUL_PIPPEN or \
                 my_kernels[0] == CB_EC2_JAC_MUL_OPT:

             if i == 0:
               mresult= ec_jacreduce_h(
                     np.reshape(input_vector[:int(len(input_vector)/5)],-1),
                     np.reshape(input_vector[int(len(input_vector)/5):],-1),
                           "".encode("UTF-8"),
                            0,
                            0,
                            2,
                            1,
                            MOD_FP, 1, 1, 1, DEFAULT_PIPPENGERS_CONF)
             #print("SCL")  
             #print(input_vector[0:8])
             #print("ECP")  
             #print(input_vector[8:])
         else:
             input_vector = tmp

         if kernel_config['kernel_idx'][0] == CB_EC2_JAC_MAD_SHFL or \
            kernel_config['kernel_idx'][0] == CB_EC2_JAC_RED:
            separate_k = 1

            if kernel_config['kernel_idx'][0] == CB_EC2_JAC_MAD_SHFL :
                separate_k = 0

            start = time.time()
            a, r,kernel_time = ec_mad_cuda2(cu_ec2bn128, input_vector, midx, 1, shamir_en=1, gpu_id=0, stream_id=0,separate_k=separate_k)
            end = time.time()

         else:
           print(kernel_params, kernel_config)
           #r,kernel_time = cu_ec2bn128.kernelLaunch(input_vector, kernel_config, kernel_params,gpu_id,n_streams,n_kernels = nkernels)
           start = time.time()
           r,kernel_time = cu_ec2bn128.kernelLaunch(input_vector, kernel_config, kernel_params,gpu_id,n_streams,n_kernels = 4)
           end = time.time()

         raff = ec2_jac2aff_h(np.reshape(r,-1), midx, 1)
         print("Proof Result" , raff)
         print("Master result", mresult)
         
         rp = np.reshape(r,(-1,6,8))
         """
         for pidx, p in enumerate(np.reshape(raff,(-1,4,8))):
          pok_aff =  ec_isoncurve_h(np.reshape(p,-1), 1,1, MOD_FP) 
          if pok_aff == 0:
             print("Is on curve : ", pok_aff, p, rp[pidx],pidx)
             break
         """
         
         if i :
             kernel_stats.append(kernel_time)
         print(end-start)
  
      
      if niter > 1:
        logging.info("Kernel %s : - Max : %s [s], Min : %s [s], Mean : %s[s]" % (k, np.max(kernel_stats), np.min(kernel_stats), np.mean(kernel_stats)))
  
      release_h()
  
if __name__ == "__main__":
    profile_ec2bn128()




