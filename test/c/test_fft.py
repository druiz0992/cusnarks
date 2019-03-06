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

nsamples = 128
n_rows = 3
n_cols = 4
prime = ZUtils.CURVE_DATA['BN128']['prime_r']
ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
ZPoly(1,force_init=True)
roots_rdc, inv_roots_rdc = ZField.find_roots(ZUtils.NROOTS, rformat_ext=False)
cu_zpoly = ZCUPoly(2*nsamples, seed=560)
u256 = U256(nsamples, seed=560)
kernel_config = {}
kernel_params = {}
zpoly_vector = cu_zpoly.rand(CUZPolyTest.nsamples)
roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc[::ZUtils.NROOTS/nsamples]])

# do mod operatio
kernel_config['gridD'] = 0
kernel_params['in_length'] = CUZPolyTest.nsamples
kernel_params['out_length'] = CUZPolyTest.nsamples
kernel_params['stride'] = 1
kernel_params['midx']=0
kernel_config['smemS'] = 0
kernel_config['blockD'] = U256_BLOCK_DIM 
zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )


p_rdc = ZPoly.from_uint256(zpoly_vector, reduced=True)
p_rdc.ntt_parallel2D(1<<n_rows,1<<n_cols)

r = ntt_parallel_h(zpoly_vector,roots_rdc_u256, n_rows, n_cols, 1)
r == p_rdc
