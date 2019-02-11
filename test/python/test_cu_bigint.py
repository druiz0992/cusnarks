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
// File name  : test_cu_bigint.py
//
// Date       : 6/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   BigInt test for CUDA
//
// TODO 
//    incorrect format  -> once asserts substituted by exceptions,
//         test incorrect formats can be done
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


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
from cu_u256 import *

sys.path.append('../../src/python')
from bigint import *

class CUBigIntTest(unittest.TestCase):
    TEST_ITER = 1000
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    nsamples = 100000
    ntest_points = 1000
    u256_p = ZField.get_extended_p().as_uint256()
    u256 = U256(u256_p, nsamples,0)

    def test_0uint256(self):
        """
        z1 =  BigInt.from_uint256(np.asarray([1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335, 719885386, 576018668]))
        z2 =  BigInt.from_uint256(np.asarray([596516649, 1189641421 ,1025202362 ,1350490027 , 783368690 ,1102520059 ,2044897763 ,893772102]))

        z1_rdc = ZFieldElRedc(z1)
        z2_rdc = ZFieldElRedc(z2)
        z12_rdc = z1_rdc * z2_rdc
        """

        # Check uint256 conversion routines provide expected results
        nsamples = 100
        u256 = CUBigIntTest.u256
        u256_vector = u256.rand(nsamples)
        bn_vector  = [BigInt.from_uint256(n) for n in u256_vector]
        r_bn = [x.as_uint256() for x in bn_vector]

        self.assertTrue(all([all(x==y) for x,y in zip(u256_vector, r_bn) ]))

    def test_1uint256(self):

        u256 = CUBigIntTest.u256
        ntest_points = CUBigIntTest.ntest_points
        u256_p = CUBigIntTest.u256_p
        for iter in xrange(CUBigIntTest.TEST_ITER):
            u256_vector = u256.rand(CUBigIntTest.nsamples)


            # Test add kernel:
            test_points = sample(xrange(CUBigIntTest.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.add(u256_vector)
            r_add, _ = BigInt.addu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)])
    
            self.assertTrue(len(result) == CUBigIntTest.nsamples/2)
            self.assertTrue(result[test_points] == r_add)
         
            # Test sub kernel:
            test_points = sample(xrange(CUBigIntTest.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.sub(u256_vector)
            r_sub, _ = BigInt.subu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)])
    
            self.assertTrue(len(result) == CUBigIntTest.nsamples/2)
            self.assertTrue(result[test_points] == r_sub)

            # Test addm kernel:
            test_points = sample(xrange(CUBigIntTest.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.addm(u256_vector)
            r_addm = BigInt.addmu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)], u256_p)
    
            self.assertTrue(len(result) == CUBigIntTest.nsamples/2)
            self.assertTrue(result[test_points] == r_addm)

            # Test subm kernel:
            test_points = sample(xrange(CUBigIntTest.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.subm(u256_vector)
            r_subm = BigInt.submu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)], u256_p)
    
            self.assertTrue(len(result) == CUBigIntTest.nsamples/2)
            self.assertTrue(result[test_points] == r_subm)

            # Test mod kernel:
            test_points = sample(xrange(CUBigIntTest.nsamples/2-2), ntest_points)

            result = u256.mod(u256_vector)
            r_mod = BigInt.modu256(u256_vector[test_points], u256_p)
    
            self.assertTrue(len(result) == CUBigIntTest.nsamples)
            self.assertTrue(result[test_points] == r_subm)

            # Test mulmont kernel:
            test_points = sample(xrange(CUBigIntTest.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.mulmont(u256_vector)
            x1_rdc = [ZFieldElRedc(BigInt.from_uint256(x) for x in u256_vector[test_points2])
            x2_rdc = [ZFieldElRedc(BigInt.from_uint256(x) for x in u256_vector[np.add(test_points2,1)])
            r_rdc = [x + y for x,y in zip(x1_rdx, x2_rdc)]
            r_mul = [x.as_uint256() for x in r_rdc]
    
            self.assertTrue(len(result) == CUBigIntTest.nsamples/2)
            self.assertTrue(result[test_points] == r_mul)

if __name__ == "__main__":
    unittest.main()
