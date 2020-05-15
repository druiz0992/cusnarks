import os,sys, os.path
import unittest
import numpy as np
from random import randint, sample

sys.path.append('../../src/python')

from bigint import *
from zutils import *
from zfield import *
from zpoly import *
from constants import *

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

ZPOLY_datafile = '../python/aux_data/zpoly_data_1M.npz'
nsamples = 1024
n_rows = int(np.log2(nsamples)/2)
n_cols = int(np.log2(nsamples / (1 << n_rows)))
prime = ZUtils.CURVE_DATA['BN128']['prime_r']
ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
ZPoly(1,force_init=True)
ZUtils.NROOTS = nsamples
if nsamples == 1024*1024 and os.path.exists(ZPOLY_datafile):
        npzfile = np.load(ZPOLY_datafile)
        roots_rdc_u256 = npzfile['roots_rdc_u256']

else:
    roots_rdc, inv_roots_rdc = ZField.find_roots(ZUtils.NROOTS, rformat_ext=False)
    roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc[::ZUtils.NROOTS / nsamples]])

cu_zpoly = ZCUPoly(2*nsamples, seed=560)
u256 = U256(nsamples, seed=560)
kernel_config = {}
kernel_params = {}
zpoly_vector = cu_zpoly.rand(nsamples)



# do mod operatio
kernel_config['gridD'] = 0
kernel_params['in_length'] = nsamples
kernel_params['out_length'] = nsamples
kernel_params['stride'] = 1
kernel_params['midx']=0
kernel_config['smemS'] = 0
kernel_config['blockD'] = U256_BLOCK_DIM 
zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )

kernel_params['in_length'] = [2 * nsamples, nsamples]
kernel_params['out_length'] = nsamples
kernel_params['stride'] = [2, 1]
kernel_params['premod'] = [0, 0]
kernel_params['midx'] = [MOD_FR, MOD_FR]
kernel_params['fft_Nx'] = [5, 5]
kernel_params['fft_Ny'] = [5, 5]
kernel_params['forward'] = [1, 1]

kernel_config['smemS'] = [0, 0]
kernel_config['blockD'] = [256, 256]
kernel_config['gridD'] = [0, (kernel_config['blockD'][0] + nsamples - 1) / kernel_config['blockD'][0]]

kernel_config['kernel_idx'] = [CB_ZPOLY_FFT2DX, CB_ZPOLY_FFT2DY]
zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))
result_fft2d, _ = cu_zpoly.kernelMultipleLaunch(zpoly_vector1, kernel_config, kernel_params, 2)


r_u256 = ntt_parallel_h(zpoly_vector,roots_rdc_u256, n_rows, n_cols, 1)

#r2_u256 = ntt_parallel2D_h(zpoly_vector,roots_rdc_u256, n_rows, 5, n_cols,5, 1)
all(np.concatenate(result_fft2d == r_u256))
