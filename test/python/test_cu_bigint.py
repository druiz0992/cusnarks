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


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
from cu_u256 import *

sys.path.append('../../src/python')
from bigint import *

class CUBigIntTest(unittest.TestCase):
    TEST_ITER = 1000
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    nsamples = 100000
    ntest_points = 1000
    u256_p = BigInt(prime).as_uint256()
    u256 = U256(u256_p, nsamples,0)

    def test_0uint256(self):

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

            # Test mod kernel:
            test_points = sample(xrange(CUBigIntTest.nsamples/2-2), ntest_points)

            result = u256.mod(u256_vector[:CUBigIntTest.nsamples/2])
            r_mod = BigInt.modu256(u256_vector[test_points], u256_p)
    
            self.assertTrue(len(result) == CUBigIntTest.nsamples/2)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_mod)))

            # Test addm kernel:
            test_points = sample(xrange(CUBigIntTest.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.addm(u256_vector)
            r_addm = BigInt.addmu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)], u256_p)
            #r_addm,_ = BigInt.addu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)])
    
            self.assertTrue(len(result) == CUBigIntTest.nsamples/2)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_addm)))

            # Test subm kernel:
            test_points = sample(xrange(CUBigIntTest.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.subm(u256_vector)
            r_subm = BigInt.submu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)], u256_p)
            #r_subm, _ = BigInt.subu256(u256_vector[test_points2], u256_vector[np.add(test_points2,1)])
    
            self.assertTrue(len(result) == CUBigIntTest.nsamples/2)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_subm)))


if __name__ == "__main__":
    unittest.main()
