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
// File name  : test_cu_zpoly.py
//
// Date       : 26/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   ZPoly CUDA modulo test
//
// ------------------------------------------------------------------

"""
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
from pycusnarks import *

sys.path.append('../../src/python')

class CUZPolyTest(unittest.TestCase):
    TEST_ITER = 1000
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    nsamples = 32 * 1024   # 1024 FFT 32 points
    ntest_points = 100 
    u256_p = BigInt(prime).as_uint256()
    cu_zpoly = ZCUPoly(nsamples, seed=560)
    u256 = U256(nsamples, seed=560)
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    ZPoly(1,force_init=True)
    roots_ext, inv_roots_ext = ZField.find_roots(ZUtils.NROOTS, rformat_ext=True)
    roots_rdc = [r.reduce() for r in roots_ext]
    inv_roots_rdc = [r.reduce() for r in inv_roots_ext]

    def test_0fft(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc

        kernel_config = {'blockD' : 256 }
        kernel_params = {'midx' : MOD_FIELD ,'premod' : 0, 'in_length' : CUZPolyTest.nsamples, 'stride' : 1, 'out_length' : CUZPolyTest.nsamples}
        # Test mod kernel:
        test_points = sample(xrange(CUZPolyTest.nsamples-1), ntest_points)



        for iter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.rand(CUZPolyTest.nsamples)
            # do mod operation
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )

            # Test FFT kernel:
            test_points = sample(xrange(CUZPolyTest.nsamples/32-1), ntest_points)
            ZField.roots[0] = roots_rdc
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_params['premod'] = 0
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = 256
            result,_ = cu_zpoly.kernelLaunch(CB_ZPOLY_FFT, zpoly_vector, kernel_config, kernel_params )
            p_rdc = ZPoly.from_uint256(zpoly_vector[0:32], reduced=True)
            p_rdc.ntt2()

            for i in test_points:
               p_rdc = ZPoly.from_uint256(zpoly_vector[i:i+32], reduced=True)
               zpoly_fft = ZPoly.from_uint256(result[i:i+32], reduced=True)
               p_rdc.ntt()
               self.assertTrue(p_rdc == zpoly_fft)


if __name__ == "__main__":
    unittest.main()
