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
// File name  : test_cu_zfield_element.py
//
// Date       : 6/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Test ZFieldl Element test for CUDA
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

class CUZFieldElTest(unittest.TestCase):
    TEST_ITER = 1000
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    nsamples = 100000
    ntest_points = 1000
    u256_p = BigInt(prime).as_uint256()
    u256 = U256(u256_p, nsamples,0)
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    reduction_data = ZField.get_reduction_data()

    def test_0mul(self):

        u256 = CUZFieldElTest.u256
        ntest_points = CUZFieldElTest.ntest_points
        u256_p = CUZFieldElTest.u256_p
        nprime = CUZFieldElTest.reduction_data['Pp'] & 0xFFFFFFFF
        for iter in xrange(CUZFieldElTest.TEST_ITER):
            u256_vector = u256.rand(CUZFieldElTest.nsamples)

            # Test mulmont kernel:
            test_points = sample(xrange(CUZFieldElTest.nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)

            result = u256.mulmont(u256_vector, nprime)
            x1_rdc = [ZFieldElRedc(BigInt.from_uint256(x)) for x in u256_vector[test_points2]]
            x2_rdc = [ZFieldElRedc(BigInt.from_uint256(x)) for x in u256_vector[np.add(test_points2,1)]]
            r_rdc = [x + y for x,y in zip(x1_rdc, x2_rdc)]
            r_mul = np.asarray([x.as_uint256() for x in r_rdc])
    
            self.assertTrue(len(result) == CUZFieldElTest.nsamples/2)
            self.assertTrue(all(np.concatenate(result[test_points]) == np.concatenate(r_mul)))

if __name__ == "__main__":
    unittest.main()
