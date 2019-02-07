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
from random import randint

sys.path.append('../../src/python')

from bigint import *
from zutils import *
from zfield import *


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
#import cu_bigint

sys.path.append('../../src/python')
from bigint import *

class CUBigIntTest(unittest.TestCase):
    TEST_ITER = 1000

    def test_0uint256(self):
        prime = ZUtils.CURVE_DATA['BN128']['prime_r']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

        z1 =  BigInt.from_uint256(np.asarray([1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335, 719885386, 576018668]))
        z2 =  BigInt.from_uint256(np.asarray([596516649, 1189641421 ,1025202362 ,1350490027 , 783368690 ,1102520059 ,2044897763 ,893772102]))




        z1_rdc = ZFieldElRedc(z1)
        z2_rdc = ZFieldElRedc(z2)
        z12_rdc = z1_rdc * z2_rdc


        for iter in xrange(CUBigIntTest.TEST_ITER):
            bn = [BigInt(randint(0,prime)) for x in range(100)]
            bn256 = [n.as_uint256() for n in bn]
            r_bn  = [BigInt.from_uint256(n) for n in bn256]

            self.assertTrue(bn == r_bn)
   
            # reduce Montgomery uint256
            zel_rdc = [ZFieldElExt(x).reduce() for x in bn]

            self.assertTrue(r1_rdc == r2_rdc)

            #bn_vector = cu_bigint.BigInt(bn256)
            #bn_vector.addm()
    
            #results2 = bn_vector.retreive()


if __name__ == "__main__":
    unittest.main()
