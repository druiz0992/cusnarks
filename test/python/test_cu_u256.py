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
    nsamples = 1024*128
    ntest_points = 10000
    u256_p = BigInt(prime).as_uint256()
    u256 = U256(nsamples, seed=560)
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
        kernel_config = {'blockD' : [U256_BLOCK_DIM] }
        kernel_params = {'midx' : [MOD_FIELD] ,'premod' : [1], 'in_length' : [CUU256Test.nsamples], 'stride' : [1], 'out_length' : CUU256Test.nsamples}
        for iter in xrange(CUU256Test.TEST_ITER):
            #if iter%CUU256Test.TEST_ITER == 0:

              ##first_sample = np.copy(u256_p)
              ##first_sample[0] -= randint(0,1000*(iter+1))
              ##u256_vector = rangeu256_h(CUU256Test.nsamples, first_sample, 100*(iter), u256_p)
              #first_sample = np.zeros(8,dtype=np.uint32)
              #u256_vector = rangeu256_h(CUU256Test.nsamples, first_sample, 1, u256_p)
            #else:
            u256_vector = u256.randu256(CUU256Test.nsamples, u256_p)
            #u256_vector = np.tile(np.ones(8,dtype=np.uint32)*44444444,(CUU256Test.nsamples,1))
            #x = u256_vector[0:2] 
            #r_mod = BigInt.modu256(x, u256_p)
            # Test mod kernel:
            #test_points = sample(xrange(CUU256Test.nsamples-1), ntest_points)

            kernel_params['in_length'] = [CUU256Test.nsamples]
            kernel_params['out_length'] = CUU256Test.nsamples
            kernel_params['stride'] = [1]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [U256_BLOCK_DIM]
            kernel_config['kernel_idx'] = [CB_U256_MOD]
            result,_ = u256.kernelLaunch(u256_vector, kernel_config, kernel_params )
            r_mod = BigInt.modu256(u256_vector, u256_p)
   
            self.assertTrue(len(result) == CUU256Test.nsamples)
            self.assertTrue(all(np.concatenate(result) == np.concatenate(r_mod)))

            #Test shr kernel:
            test_points = sample(xrange(CUU256Test.nsamples-1), ntest_points)

            kernel_params['in_length'] = [CUU256Test.nsamples]
            kernel_params['out_length'] = CUU256Test.nsamples
            kernel_params['stride'] = [1]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [U256_BLOCK_DIM]
            kernel_config['kernel_idx']=[CB_U256_SHR1]
            result,_ = u256.kernelLaunch(u256_vector, kernel_config, kernel_params )
            r_shr = u256_vector[test_points]
    
            self.assertTrue(len(result) == CUU256Test.nsamples)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_shr)))

            #Test shl kernel:
            test_points = sample(xrange(CUU256Test.nsamples-1), ntest_points)

            kernel_params['in_length'] = [CUU256Test.nsamples]
            kernel_params['out_length'] = CUU256Test.nsamples
            kernel_params['stride'] = [1]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [U256_BLOCK_DIM]
            kernel_config['kernel_idx']=[CB_U256_SHL1]
            result,_ = u256.kernelLaunch(u256_vector, kernel_config, kernel_params )
            r_shl = u256_vector[test_points]
    
            self.assertTrue(len(result) == CUU256Test.nsamples)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_shl)))
            # Test addm kernel:
            test_points = sample(xrange(CUU256Test.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            kernel_params['in_length'] = [CUU256Test.nsamples]
            kernel_params['out_length'] = CUU256Test.nsamples/2
            kernel_params['stride'] = [2]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [U256_BLOCK_DIM]
            kernel_config['kernel_idx'] = [CB_U256_ADDM]
            result,_ = u256.kernelLaunch(u256_vector, kernel_config, kernel_params )
            r_addm = BigInt.addmu256(u256_vector[::2], u256_vector[1::2], u256_p)
   
            self.assertTrue(len(result) == CUU256Test.nsamples/2)
            self.assertTrue(all(np.concatenate(result) == np.concatenate(r_addm)))

            # Test subm kernel:
            kernel_params['in_length'] = [CUU256Test.nsamples]
            kernel_params['out_length'] = CUU256Test.nsamples/2
            kernel_params['stride'] = [2]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [U256_BLOCK_DIM]
            kernel_config['kernel_idx'] = [CB_U256_SUBM]
            test_points = sample(xrange(CUU256Test.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result,_ = u256.kernelLaunch(u256_vector, kernel_config, kernel_params )
            r_subm = BigInt.submu256(u256_vector[::2], u256_vector[1::2], u256_p)
    
            self.assertTrue(len(result) == CUU256Test.nsamples/2)
            self.assertTrue(all(np.concatenate(result) == np.concatenate(r_subm)))
            
            # Test mulmont kernel:
            kernel_params['in_length'] = [CUU256Test.nsamples]
            kernel_params['out_length'] = CUU256Test.nsamples/2
            kernel_params['stride'] = [2]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [U256_BLOCK_DIM]
            kernel_config['kernel_idx'] = [CB_U256_MULM]
            #u256_vector[0] = np.asarray([1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335, 719885386, 576018668], dtype=np.uint32)
            #u256_vector[1] = np.asarray([596516649, 1189641421, 1025202362, 1350490027, 783368690, 1102520059, 2044897763, 803772102], dtype=np.uint32)
            #kernel_config = {'blockD' : 1, 'gridD' : 1 }
            #kernel_params['length'] = CUU256Test.nsamples/2
            #kernel_params['stride'] = 2
            test_points = sample(xrange(CUU256Test.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            #u256_vector[0] = np.asarray([1895965571, 2509118507, 1177768607,   19354091,         0,         0,
               #0,         0], dtype=np.uint32)
            #u256_vector[1] = np.asarray([4197368849, 3466883767, 0,       0,          0,
                #0,          0,          0], dtype=np.uint32)
            #test_points2[0]=0
            result, _ = u256.kernelLaunch(u256_vector, kernel_config, kernel_params )
            x1_rdc = [ZFieldElRedc(BigInt.from_uint256(x)) for x in u256_vector[test_points2]]
            x2_rdc = [ZFieldElRedc(BigInt.from_uint256(x)) for x in u256_vector[np.add(test_points2,1)]]
            r_rdc = [x * y for x,y in zip(x1_rdc, x2_rdc)]
            r_mul = np.asarray([x.as_uint256() for x in r_rdc])
            
            idx=0
            result2 = np.zeros(r_mul.shape, dtype=np.uint32)
            for x1,x2 in zip(x1_rdc, x2_rdc):
                result2[idx] = montmult_h(x1.as_uint256(), x2.as_uint256(), 1)
                idx+=1
            self.assertTrue(len(result) == CUU256Test.nsamples/2)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_mul)))
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(result2)))

            # Test addm_reduce kernel:
            # First iteration : Reduce by blockSize * stride => 
            r_addm_reduce = BigInt.addmu256_reduce(u256_vector[::2], u256_vector[1::2], u256_p)

            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD] 
            kernel_params['premod'] = [1, 0]
            kernel_params['stride'] = [4, 2]
            kernel_config['blockD'] = [64, 256]
            kernel_params['in_length'] = [CUU256Test.nsamples, \
                                         (CUU256Test.nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]) -1) / (kernel_config['blockD'][0]*kernel_params['stride'][0])]
            kernel_params['out_length'] = 1
            kernel_config['smemS'] = [kernel_config['blockD'][0] * NWORDS_256BIT * 4, \
                                     kernel_config['blockD'][1] * NWORDS_256BIT * 4]
            kernel_config['gridD'] = [0,  1]
            kernel_config['kernel_idx'] = [CB_U256_ADDM_REDUCE, CB_U256_ADDM_REDUCE]

            #para with zeros if necessary
            min_length = kernel_config['blockD'][1] * kernel_params['stride'][1]
            if kernel_params['in_length'][1] < min_length:
              zeros = np.zeros((min_length - kernel_params['in_length'][1],NWORDS_256BIT), dtype=np.uint32)
              result = np.concatenate((result,zeros))
              kernel_params['in_length'][1] = min_length

            result,_ = u256.kernelLaunch(u256_vector, kernel_config, kernel_params,2 )

            """
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['premod'] = [1]
            kernel_params['stride'] = [4]
            kernel_config['blockD'] = [64]
            kernel_params['in_length'] = [CUU256Test.nsamples]
            kernel_params['out_length'] = 512
            kernel_config['smemS'] = [kernel_config['blockD'][0] * NWORDS_256BIT * 4]
            kernel_config['kernel_idx'] = [CB_U256_ADDM_REDUCE]

            result,_ = u256.kernelLaunch(u256_vector, kernel_config, kernel_params,1 )


            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['premod'] = [0]
            kernel_params['stride'] = [2]
            kernel_config['blockD'] = [256]
            kernel_params['in_length'] = [512]
            kernel_params['out_length'] = 1
            kernel_config['smemS'] = [kernel_config['blockD'][0] * NWORDS_256BIT * 4]
            kernel_config['kernel_idx'] = [CB_U256_ADDM_REDUCE]

            result,_ = u256.kernelLaunch(result, kernel_config, kernel_params,1 )
            """
           
            #debugReduceAddm(u256_vector,result)
      
            self.assertTrue(len(result) == kernel_params['out_length'])
            if all(np.concatenate(result == r_addm_reduce)):
                self.assertTrue(all(np.concatenate(result == r_addm_reduce)))
            else:
               print all(np.concatenate(result == r_addm_reduce))
           

            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD] 
            kernel_params['premod'] = [1, 0]
            kernel_params['premul'] = [1, 0]
            kernel_params['stride'] = [1, 1]
            kernel_config['blockD'] = [1024, 128]
            kernel_params['in_length'] = [CUU256Test.nsamples, \
                                         (CUU256Test.nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]) -1) / (kernel_config['blockD'][0]*kernel_params['stride'][0])]
            kernel_params['out_length'] = 1
            kernel_config['smemS'] = [kernel_config['blockD'][0]/32 * NWORDS_256BIT * 4, \
                                     kernel_config['blockD'][1]/32 * NWORDS_256BIT * 4]
            kernel_config['gridD'] = [0,  1]
            kernel_config['kernel_idx'] = [CB_U256_ADDM_REDUCE_SHFL, CB_U256_ADDM_REDUCE_SHFL]

            result2,_ = u256.kernelLaunch(u256_vector, kernel_config, kernel_params,2 )

def debugReduceAddm(u256_vector, result):
   xl = [BigInt.from_uint256(x).as_long() % CUU256Test.prime for x in u256_vector]
   x_u256 = [BigInt(x).as_uint256() for x in xl]


   x1_l = [(xl[4*k]+xl[4*k+1]+xl[4*k+2]+xl[4*k+3]) %  CUU256Test.prime for k in range(len(xl)/4)]
   x1_u256 = [BigInt(x).as_uint256() for x in x1_l]

   x2_l=[]
   for i in xrange(len(x1_l)/64):
        x2_l = x2_l + [(x1_l[64*i+k]+x1_l[64*i + k+32]) %  CUU256Test.prime for k in range(32)]
   x2_u256 = [BigInt(x).as_uint256() for x in x2_l]

   x3_l = []
   for i in xrange(len(x2_l)/32):
      x3_l = x3_l +  [(x2_l[32*i+k]+x2_l[32*i+k+16]) %  CUU256Test.prime for k in range(16)]
   x3_u256 = [BigInt(x).as_uint256() for x in x3_l]

   x4_l = []
   for i in xrange(len(x3_l)/16):
      x4_l = x4_l + [(x3_l[16*i+k]+x3_l[16*i+k+8]) %  CUU256Test.prime for k in range(8)]
   x4_u256 = [BigInt(x).as_uint256() for x in x4_l]

   x5_l = []
   for i in xrange(len(x4_l)/8):
      x5_l = x5_l + [(x4_l[8*i+k]+x4_l[8*i+k+4]) %  CUU256Test.prime for k in range(4)]
   x5_u256 = [BigInt(x).as_uint256() for x in x5_l]

   x6_l = []
   for i in xrange(len(x5_l)/4):
      x6_l = x6_l + [(x5_l[4*i+k]+x5_l[4*i+k+2]) %  CUU256Test.prime for k in range(2)]
   x6_u256 = [BigInt(x).as_uint256() for x in x6_l]

   x7_l = []
   for i in xrange(len(x6_l)/2):
       x7_l = x7_l + [(x6_l[2*i+k]+x6_l[2*i+k+1]) %  CUU256Test.prime for k in range(1)]
   x7_u256 = [BigInt(x).as_uint256() for x in x7_l]

   x8_l = [(x7_l[2*k]+x7_l[2*k+1]) %  CUU256Test.prime for k in range(len(x7_l)/2)]
   x8_u256 = [BigInt(x).as_uint256() for x in x8_l]

   x9_l = [(x8_l[k]+x8_l[k+128]) %  CUU256Test.prime for k in range(128)]
   x9_u256 = [BigInt(x).as_uint256() for x in x9_l]

   x10_l = [(x9_l[k]+x9_l[k+64]) %  CUU256Test.prime for k in range(64)]
   x10_u256 = [BigInt(x).as_uint256() for x in x10_l]

   x11_l = [(x10_l[k]+x10_l[k+32]) %  CUU256Test.prime for k in range(32)]
   x11_u256 = [BigInt(x).as_uint256() for x in x11_l]

   x12_l = [(x11_l[k]+x11_l[k+16]) %  CUU256Test.prime for k in range(16)]
   x12_u256 = [BigInt(x).as_uint256() for x in x12_l]

   x13_l = [(x12_l[k]+x12_l[k+8]) %  CUU256Test.prime for k in range(8)]
   x13_u256 = [BigInt(x).as_uint256() for x in x13_l]

   x14_l = [(x13_l[k]+x13_l[k+4]) %  CUU256Test.prime for k in range(4)]
   x14_u256 = [BigInt(x).as_uint256() for x in x14_l]

   x15_l = [(x14_l[k] + x14_l[k + 2]) % CUU256Test.prime for k in range(2)]
   x15_u256 = [BigInt(x).as_uint256() for x in x15_l]

   x16_l = [(x15_l[k] + x15_l[k + 1]) % CUU256Test.prime for k in range(1)]
   x16_u256 = [BigInt(x).as_uint256() for x in x16_l]



if __name__ == "__main__":
    unittest.main()
