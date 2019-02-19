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
// File name  : test_bigint.py
//
// Date       : 15/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//   BigInt test
//
// TODO 
//    incorrect format  -> once asserts substituted by exceptions,
//         test incorrect formats can be done
// ------------------------------------------------------------------

"""
import sys
import unittest
from random import randint

sys.path.append('../../src/python')

from bigint import *
from zutils import *


class BigIntTest(unittest.TestCase):
    TEST_ITER = 1000
    LONG_THR = 23432043289577921739247892346893264783264382764L
    INT_THR = 134
    BIT_THR = randint(0, 100)

    def test_0init(self):

        ## Init given number as dec string
        a_str = "12345643245354635394758934758937895378957"
        a_bn = BigInt(a_str)
        self.assertTrue(long(a_str) == a_bn.as_long())

        ## Init given number as hex string
        a_str = "0x23452abc234590aacf67f3"
        a_bn = BigInt(a_str)
        self.assertTrue(long(a_str, 16) == a_bn.as_long())

        ## Init given number as int
        a_int = randint(-12332424, 12334545)
        a_bn = BigInt(a_int)
        self.assertTrue(long(a_int) == a_bn.as_long())

        ## Init given number as long
        a_int = randint(-13139837538927589327648264782, 1233535389758937217892748923789372434545)
        a_bn = BigInt(a_int)
        self.assertTrue(long(a_int) == a_bn.as_long())

        ## Init given number as bignum
        a_int = randint(-1239827489237498327789326785362, 1233535389758937217892748923789372434545)
        a_bn = BigInt(a_int)
        b_bn = BigInt(a_bn)
        self.assertTrue(a_bn.as_long() == b_bn.as_long())

        ## Init random number as dec string
        min_str = "1234565394758934758937895378957"
        max_str = "12345643245354635394758934758937895378957"
        bn = BigInt(max_str, min_bignum=min_str)
        self.assertTrue(bn.as_long() <= long(max_str))
        self.assertTrue(bn.as_long() >= long(min_str))

        ## Init random number as hex string
        min_str = "0xabc32546afcda34"
        max_str = "0x123dfabcf464aaccdd3256"
        bn = BigInt(max_str, min_bignum=min_str)
        self.assertTrue(bn.as_long() <= long(max_str, 16))
        self.assertTrue(bn.as_long() >= long(min_str, 16))

        ## Init random number as int
        min_int = 1324242
        max_int = 20403953495
        bn = BigInt(max_int, min_bignum=min_int)
        self.assertTrue(bn.as_long() <= long(max_int))
        self.assertTrue(bn.as_long() >= long(min_int))

        ## Init random number as long
        min_int = 13242424039340957389247
        max_int = 2040395349528479238479823472947
        bn = BigInt(max_int, min_bignum=min_int)
        self.assertTrue(bn.as_long() <= long(max_int))
        self.assertTrue(bn.as_long() >= long(min_int))

        ## Init random number as bignum
        min_bn = BigInt(13242424039340957389247)
        max_bn = BigInt(2040395349528479238479823472947)
        bn = BigInt(max_bn, min_bignum=min_bn)
        self.assertTrue(bn.as_long() <= max_bn.as_long())
        self.assertTrue(bn.as_long() >= min_bn.as_long())

    def test_1arithmetic(self):
        for i in xrange(BigIntTest.TEST_ITER):
            # +
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r_l = x_l + y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn + y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # -
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r_l = x_l - y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn - y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # *
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r_l = x_l * y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn * y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # //
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r_l = x_l // y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn // y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # pow
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(0, BigIntTest.INT_THR)
            r_l = x_l ** y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn ** y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # += 
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r_l = x_l
            r_l += y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn
            r_bn += y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # -= 
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r_l = x_l
            r_l -= y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn
            r_bn -= y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # neg
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r_l = -x_l

            x_bn = BigInt(x_l)
            r_bn = -x_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

    def test_2bitwise(self):
        for i in xrange(BigIntTest.TEST_ITER):
            # <<
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(0, BigIntTest.BIT_THR)
            r_l = x_l << y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn << y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # >>
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(0, BigIntTest.BIT_THR)
            r_l = x_l >> y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn >> y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # &
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(0, BigIntTest.BIT_THR)
            r_l = x_l & y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn & y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # |
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(0, BigIntTest.BIT_THR)
            r_l = x_l | y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn | y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # <<= 
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(0, BigIntTest.BIT_THR)
            r_l = x_l
            r_l <<= y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn
            r_bn <<= y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

            # >>= 
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(0, BigIntTest.BIT_THR)
            r_l = x_l
            r_l >>= y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r_bn = x_bn
            r_bn >>= y_bn

            self.assertTrue(r_bn.as_long() == long(r_l))

    def test_3comparison(self):
        for i in xrange(BigIntTest.TEST_ITER):
            # <
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r1 = x_l < y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r2 = x_bn < y_bn

            self.assertTrue(r1 == r2)

            # <=
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r1 = x_l <= y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r2 = x_bn <= y_bn

            self.assertTrue(r1 == r2)

            # >
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r1 = x_l > y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r2 = x_bn > y_bn

            self.assertTrue(r1 == r2)

            # >=
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            r1 = x_l >= y_l

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r2 = x_bn >= y_bn

            self.assertTrue(r1 == r2)

            # ==
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = x_l + 2

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r1 = x_bn == x_bn
            r2 = x_bn == y_bn

            self.assertTrue(r1 == True)
            self.assertTrue(r2 == False)

            # !=
            x_l = randint(-BigIntTest.LONG_THR, BigIntTest.LONG_THR)
            y_l = x_l + 2

            x_bn = BigInt(x_l)
            y_bn = BigInt(y_l)
            r1 = x_bn != x_bn
            r2 = x_bn != y_bn

            self.assertTrue(r1 == False)
            self.assertTrue(r2 == True)

    def test_4u256(self):
        prime = ZUtils.CURVE_DATA['BN128']['prime_r']
        nsamples = 1000
        p_u256 = BigInt(prime).as_uint256()

        for iter in xrange(BigIntTest.TEST_ITER):
           x_l = [randint(0, (1<<256)-1) for  x in range(nsamples)]
           y_l = [randint(0, (1<<256)-1) for x in  range(nsamples)]

           x_bn = [BigInt(x) for x in x_l]
           y_bn = [BigInt(x) for x in y_l]

           x_u256 = np.asarray([x.as_uint256() for x in x_bn])
           y_u256 = np.asarray([x.as_uint256() for x in y_bn])

           # sum
           z_bn = [(x+y) & ((1<<256)-1) for x,y in zip(x_bn,y_bn)]

           z_u256, _ = BigInt.addu256(x_u256, y_u256)
           z2_bn = [BigInt.from_uint256(x) for x in z_u256]

           self.assertTrue(z_bn == z2_bn)

           # sub
           z_bn = [(x-y) & ((1<<256)-1) for x,y in zip(x_bn,y_bn)]

           z_u256, _ = BigInt.subu256(x_u256, y_u256)
           z2_bn = [BigInt.from_uint256(x) for x in z_u256]

           self.assertTrue(z_bn == z2_bn)

           # mod
           z_bn = [x % prime for x in x_bn]

           z_u256 = BigInt.modu256(x_u256, p_u256)
           z2_bn = [BigInt.from_uint256(x) for x in z_u256]

           self.assertTrue(z_bn == z2_bn)

           # subm
           z_bn = [(x-y) % prime for x,y in zip(x_bn,y_bn)]

           z_u256 = BigInt.submu256(x_u256, y_u256, p_u256)
           z2_bn = [BigInt.from_uint256(x) for x in z_u256]

           self.assertTrue(z_bn == z2_bn)

           # addm
           z_bn = [(x+y) % prime for x,y in zip(x_bn,y_bn)]

           z_u256 = BigInt.addmu256(x_u256, y_u256, p_u256)
           z2_bn = [BigInt.from_uint256(x) for x in z_u256]

           self.assertTrue(z_bn == z2_bn)

    def test_5u256_special(self):
        prime = ZUtils.CURVE_DATA['BN128']['prime_r']
        p_u256 = BigInt(prime).as_uint256()
        p1_u256 = BigInt(prime + 1).as_uint256()
        p0_u256 = BigInt(prime - 1).as_uint256()

        x0_l = 0
        x1_l = 1
        xp_l = prime
        xp1_l = prime + 1
        xp0_l = prime - 1


        x_l = [randint(0, (1<<256)-1) % xp_l for x in  range(8)]
        y_l = [randint(0, (1<<256)-1) % xp_l for x in  range(8)]

        x0_bn = BigInt(x0_l)
        x1_bn = BigInt(x1_l)
        xp_bn = BigInt(xp_l)
        x_bn = [BigInt(x) for x in x_l]
        y_bn = [BigInt(x) for x in y_l]

        x0_u256 = np.tile(np.asarray(x0_bn.as_uint256()),(len(x_l),1))
        x1_u256 = np.tile(np.asarray(x1_bn.as_uint256()),(len(x_l),1))
        xp_u256 = np.tile(np.asarray(xp_bn.as_uint256()),(len(x_l),1))

        x0_bn = np.tile(x0_bn, len(x_l))
        x1_bn = np.tile(x1_bn, len(x_l))
        xp_bn = np.tile(xp_bn, len(x_l))

        x_u256 = np.asarray([x.as_uint256() for x in x_bn])
        y_u256 = np.asarray([x.as_uint256() for x in y_bn])

        # sum 0 + y
        z_bn = [(x+y) & ((1<<256)-1) for x,y in zip(x0_bn,y_bn)]

        z_u256, _ = BigInt.addu256(x0_u256, y_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sum x + 0
        z_bn = [(x+y) & ((1<<256)-1) for x,y in zip(x_bn,x0_bn)]

        z_u256, _ = BigInt.addu256(x_u256, x0_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sum p
        z_bn = [(x+y) & ((1<<256)-1) for x,y in zip(xp_bn,y_bn)]

        z_u256, _ = BigInt.addu256(xp_u256, y_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)


        # sub 0 - y
        z_bn = [(x-y) & ((1<<256)-1) for x,y in zip(x0_bn,y_bn)]

        z_u256, _ = BigInt.subu256(x0_u256, y_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub x - 0
        z_bn = [(x-y) & ((1<<256)-1) for x,y in zip(x_bn,x0_bn)]

        z_u256, _ = BigInt.subu256(x_u256, x0_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub x - p
        z_bn = [(x-y) & ((1<<256)-1) for x,y in zip(x_bn,xp_bn)]

        z_u256, _ = BigInt.subu256(x_u256, xp_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub p - y
        z_bn = [(x-y) & ((1<<256)-1) for x,y in zip(xp_bn,x_bn)]

        z_u256, _ = BigInt.subu256(xp_u256, x_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub x - x
        z_bn = [(x-y) & ((1<<256)-1) for x,y in zip(x_bn,x_bn)]

        z_u256, _ = BigInt.subu256(x_u256, x_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sum 0 + y % p
        z_bn = [(x+y) for x,y in zip(x0_bn,y_bn)]

        z_u256 = BigInt.addmu256(x0_u256, y_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sum x + 0 % p
        z_bn = [(x+y) % xp_l for x,y in zip(x_bn,x0_bn)]

        z_u256 = BigInt.addmu256(x_u256, x0_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sum p + y % p
        z_bn = [(x+y) % xp_l for x,y in zip(xp_bn,y_bn)]

        z_u256 = BigInt.addmu256(xp_u256, y_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub 0 - y % p
        z_bn = [(x-y) % xp_l for x,y in zip(x0_bn,y_bn)]

        z_u256= BigInt.submu256(x0_u256, y_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub x - 0 % p
        z_bn = [(x-y)  % xp_l for x,y in zip(x_bn,x0_bn)]

        z_u256 = BigInt.submu256(x_u256, x0_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub x - p % p
        z_bn = [(x-y) %xp_l for x,y in zip(x_bn,xp_bn)]

        z_u256 = BigInt.submu256(x_u256, xp_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub p - y % p
        z_bn = [(x-y) % xp_l for x,y in zip(xp_bn,x_bn)]

        z_u256 = BigInt.submu256(xp_u256, x_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub x - x % p
        z_bn = [(x-y) % xp_l for x,y in zip(x_bn,x_bn)]

        z_u256 = BigInt.submu256(x_u256, x_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub x1 - x % p
        z_bn = [(x-y)  % xp_l for x,y in zip(x1_bn,y_bn)]

        z_u256 = BigInt.submu256(x1_u256, y_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

        # sub x - x1 % p
        z_bn = [(x-y) % xp_l for x,y in zip(x_bn,x1_bn)]

        z_u256 = BigInt.submu256(x_u256, x1_u256, p_u256)
        z2_bn = [BigInt.from_uint256(x) for x in z_u256]

        self.assertTrue(z_bn == z2_bn)

           
            

if __name__ == "__main__":
    unittest.main()
