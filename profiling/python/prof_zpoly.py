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
// File name  : prof_zpoly.py
//
// Date       : 13/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Profile zpoly CUDA Module
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

logging.basicConfig(level=logging.DEBUG,
                    format='[%(processName)s] %(asctime)s %(levelname)s %(message)s')


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
from pycusnarks import *

sys.path.append('../../src/python')
ZPOLY_datafile = '../../data/zpoly_data_1M.npz'

def profile_zpoly_1024():

    niter = 100
    kernel_stats = []
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    ZPoly(1,force_init=True)
    roots_ext, inv_roots_ext = ZField.find_roots(ZUtils.NROOTS, rformat_ext=True)
    roots_rdc = [r.reduce() for r in roots_ext]
    inv_roots_rdc = [r.reduce() for r in inv_roots_ext]
    nsamples = 1<<10
    cu_zpoly = ZCUPoly(2*nsamples, seed=560)

    kernel_params = {}
    kernel_config = {}


    u256 = U256(nsamples, seed=10)
    zpoly_vector = cu_zpoly.rand(nsamples)

    # do mod operatio
    kernel_config['gridD'] = [0]
    kernel_params['in_length'] = [nsamples]
    kernel_params['out_length'] = nsamples
    kernel_params['stride'] = [1]
    kernel_params['midx']=[0]
    kernel_config['smemS'] = [0]
    kernel_config['blockD'] = [U256_BLOCK_DIM ]  
    kernel_config['kernel_idx'] = [CB_U256_MOD]
    zpoly_vector,_ = u256.kernelLaunch(zpoly_vector, kernel_config, kernel_params )

    # Test FFT kernel:
    kernel_params['in_length'] = [2*nsamples,nsamples]
    kernel_params['out_length'] = nsamples
    kernel_params['stride'] = [2,1]
    kernel_params['premod'] = [0,0]
    kernel_params['midx'] = [MOD_FIELD, MOD_FIELD]
    kernel_params['fft_Nx'] = [5,5]
    kernel_params['fft_Ny'] = [5,5]
    kernel_params['forward'] = [1,1]
    roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc[::ZUtils.NROOTS/1024]])

    kernel_config['smemS'] = [0,0]
    kernel_config['blockD'] = [256,256]
    kernel_config['gridD'] = [0, (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
    kernel_config['kernel_idx']= [CB_ZPOLY_FFT2DX, CB_ZPOLY_FFT2DY]
    zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))


    for i in range(niter):
       _,kernel_time = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,n_kernels = 2)
       if i :
           kernel_stats.append(kernel_time)

    
    logging.info("Max : %s [s], Min : %s [s], Mean : %s[s]" % (np.max(kernel_stats), np.min(kernel_stats), np.mean(kernel_stats)))


def profile_zpoly_1M():

    niter = 20
    kernel_stats = []
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    ZPoly(1,force_init=True)
    roots_ext, inv_roots_ext = ZField.find_roots(ZUtils.NROOTS, rformat_ext=True)
    roots_rdc = [r.reduce() for r in roots_ext]
    inv_roots_rdc = [r.reduce() for r in inv_roots_ext]
    nsamples = 1<<20
    cu_zpoly = ZCUPoly(2*nsamples, seed=560)

    kernel_params = {}
    kernel_config = {}

    if os.path.exists(ZPOLY_datafile):
         npzfile = np.load(ZPOLY_datafile)
         roots_rdc_u256 = npzfile['roots_rdc_u256']
       
    else:
      print ("Error. File not found")
      return
          

    u256 = U256(nsamples, seed=10)
    zpoly_vector = cu_zpoly.rand(nsamples)
    n_rows=10
    n_cols = 10
    fft_N = 5

    # do mod operatio
    kernel_config['gridD'] = [0]
    kernel_params['in_length'] = [nsamples]
    kernel_params['out_length'] = nsamples
    kernel_params['stride'] = [1]
    kernel_params['midx']=[0]
    kernel_config['smemS'] = [0]
    kernel_config['blockD'] = [U256_BLOCK_DIM ]  
    kernel_config['kernel_idx'] = [CB_U256_MOD]
    zpoly_vector,_ = u256.kernelLaunch(zpoly_vector, kernel_config, kernel_params )

    # Test FFT kernel:
    kernel_params['in_length'] = [2*nsamples, nsamples, nsamples, nsamples]
    kernel_params['out_length'] = nsamples
    kernel_params['stride'] = [2,1,1,1]
    kernel_params['premod'] = [0,0,0,0]
    kernel_params['midx'] = [MOD_FIELD, MOD_FIELD, MOD_FIELD, MOD_FIELD]
    kernel_params['fft_Nx'] = [fft_N, fft_N, fft_N, fft_N]
    kernel_params['fft_Ny'] = [fft_N, fft_N, fft_N, fft_N]
    kernel_params['N_fftx'] = [n_cols, n_cols, n_cols, n_cols]
    kernel_params['N_ffty'] = [n_rows, n_rows, n_rows, n_rows]
    kernel_params['forward'] = [1, 1, 1, 1]

    kernel_config['smemS'] = [0, 0, 0, 0]
    kernel_config['blockD'] = [256, 256, 256, 256]
    kernel_config['gridD'] = [0,  (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
    kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]

    zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))


    for i in range(niter):
       _,kernel_time = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,n_kernels = 4)
       if i :
           kernel_stats.append(kernel_time)

    
    logging.info("Max : %s [s], Min : %s [s], Mean : %s[s]" % (np.max(kernel_stats), np.min(kernel_stats), np.mean(kernel_stats)))

if __name__ == "__main__":
    profile_zpoly_1M()
