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
// File name  : test_cu_u256.py
//
// Date       : 6/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   U256 CUDA modulo test
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
from constants import *


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
from pycusnarks import *

sys.path.append('../../src/python')
from bigint import *

class CUU256Test(unittest.TestCase):
    TEST_ITER = 1000
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    nsamples = 141072
    nsamples = 1024*128
    ntest_points = 1000
    u256_p = BigInt(prime).as_uint256()
    u256 = U256(nsamples, seed=10)
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

    def test_0uint256(self):

        # Check uint256 conversion routines provide expected results
        nsamples = 100
        u256 = CUU256Test.u256
        u256_vector = u256.rand(nsamples)
        bn_vector  = [BigInt.from_uint256(n) for n in u256_vector]
        r_bn = [x.as_uint256() for x in bn_vector]

        self.assertTrue(all([all(x==y) for x,y in zip(u256_vector, r_bn) ]))

    def test_1uint256(self):

        u256 = CUU256Test.u256
        ntest_points = CUU256Test.ntest_points
        u256_p = CUU256Test.u256_p
        kernel_config = {'blockD' : U256_BLOCK_DIM }
        kernel_params = {'midx' : MOD_FIELD ,'premod' : 1, 'in_length' : CUU256Test.nsamples, 'stride' : 1, 'out_length' : CUU256Test.nsamples}
        for iter in xrange(CUU256Test.TEST_ITER):
            u256_vector = u256.rand(CUU256Test.nsamples)
            #u256_vector = np.tile(np.ones(8,dtype=np.uint32)*44444444,(CUU256Test.nsamples,1))
            
            # Test mod kernel:
            test_points = sample(xrange(CUU256Test.nsamples-1), ntest_points)

            kernel_params['in_length'] = CUU256Test.nsamples
            kernel_params['out_length'] = CUU256Test.nsamples
            kernel_params['stride'] = 1
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            result = u256.kernelLaunch(CB_U256_MOD, u256_vector, kernel_config, kernel_params )
            r_mod = BigInt.modu256(u256_vector[test_points], u256_p)
    
            self.assertTrue(len(result) == CUU256Test.nsamples)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_mod)))

            #Test shl kernel:
            test_points = sample(xrange(CUU256Test.nsamples-1), ntest_points)

            kernel_params['in_length'] = CUU256Test.nsamples
            kernel_params['out_length'] = CUU256Test.nsamples
            kernel_params['stride'] = 1
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            result = u256.kernelLaunch(CB_U256_SHL1, u256_vector, kernel_config, kernel_params )
            r_shl = u256_vector[test_points]
    
            self.assertTrue(len(result) == CUU256Test.nsamples)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_shl)))
            # Test addm kernel:
            test_points = sample(xrange(CUU256Test.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            kernel_params['in_length'] = CUU256Test.nsamples
            kernel_params['out_length'] = CUU256Test.nsamples/2
            kernel_params['stride'] = 2
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            result = u256.kernelLaunch(CB_U256_ADDM, u256_vector, kernel_config, kernel_params )
            r_addm = BigInt.addmu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)], u256_p)
    
            self.assertTrue(len(result) == CUU256Test.nsamples/2)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_addm)))

            # Test subm kernel:
            kernel_params['in_length'] = CUU256Test.nsamples
            kernel_params['out_length'] = CUU256Test.nsamples/2
            kernel_params['stride'] = 2
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            test_points = sample(xrange(CUU256Test.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.kernelLaunch(CB_U256_SUBM, u256_vector, kernel_config, kernel_params )
            r_subm = BigInt.submu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)], u256_p)
    
            self.assertTrue(len(result) == CUU256Test.nsamples/2)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_subm)))
            
            # Test mulmont kernel:
            kernel_params['in_length'] = CUU256Test.nsamples
            kernel_params['out_length'] = CUU256Test.nsamples/2
            kernel_params['stride'] = 2
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            #u256_vector[0] = np.asarray([1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335, 719885386, 576018668], dtype=np.uint32)
            #u256_vector[1] = np.asarray([596516649, 1189641421, 1025202362, 1350490027, 783368690, 1102520059, 2044897763, 803772102], dtype=np.uint32)
            #kernel_config = {'blockD' : 1, 'gridD' : 1 }
            #kernel_params['length'] = CUU256Test.nsamples/2
            #kernel_params['stride'] = 2
            test_points = sample(xrange(CUU256Test.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.kernelLaunch(CB_U256_MULM, u256_vector, kernel_config, kernel_params )
            x1_rdc = [ZFieldElRedc(BigInt.from_uint256(x)) for x in u256_vector[test_points2]]
            x2_rdc = [ZFieldElRedc(BigInt.from_uint256(x)) for x in u256_vector[np.add(test_points2,1)]]
            r_rdc = [x * y for x,y in zip(x1_rdc, x2_rdc)]
            r_mul = np.asarray([x.as_uint256() for x in r_rdc])
    
            self.assertTrue(len(result) == CUU256Test.nsamples/2)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_mul)))

            # Test addm_reduce kernel:
            # First iteration : Reduce by blockSize * stride => 
            r_addm_reduce = BigInt.addmu256_reduce(u256_vector[::2], u256_vector[1::2], u256_p)

            kernel_params['in_length'] = CUU256Test.nsamples
            kernel_params['stride'] = 4
            kernel_config['blockD'] = U256_BLOCK_DIM 
            kernel_params['out_length'] = (CUU256Test.nsamples + (kernel_config['blockD']*kernel_params['stride']) -1) / (kernel_config['blockD']*kernel_params['stride'])
            kernel_config['smemS'] = kernel_config['blockD'] * NWORDS_256BIT * 4 
            result = u256.kernelLaunch(CB_U256_ADDM_REDUCE, u256_vector, kernel_config, kernel_params )

            kernel_params['in_length'] = kernel_params['out_length']
            kernel_params['stride'] = 2
            kernel_params['out_length'] = 1
            kernel_config['blockD'] = 64
            kernel_config['smemS'] = kernel_config['blockD'] * NWORDS_256BIT * 4 
            min_length = kernel_config['blockD'] * kernel_params['stride']
            if kernel_params['in_length'] < min_length:
               zeros = np.zeros((min_length - kernel_params['in_length'],NWORDS_256BIT), dtype=np.uint32)
               result = np.concatenate((result,zeros))
               kernel_params['in_length'] = min_length

            result = u256.kernelLaunch(CB_U256_ADDM_REDUCE, result, kernel_config, kernel_params )
    
            self.assertTrue(len(result) == kernel_params['out_length'])
            self.assertTrue(all(np.concatenate(result == r_addm_reduce)))


if __name__ == "__main__":
    unittest.main()
