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
// File name  : test_zfield_element.py
//
// Date       : 18/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//   ZField Element test
//
// TODO 
//    incorrect format  -> once asserts substituted by exceptions,
//         test incorrect formats can be done
// ------------------------------------------------------------------

"""
from random import randint
import unittest

import sys
sys.path.append('../../src/python')

from bigint import *
from zfield import *
from zutils import *

class ZFieldElTest(unittest.TestCase):

    TEST_ITER = 1000
    INT_THR  = 13423
    LONG_THR = ZField.FIELD_DATA['BN128']['prime'] * 4
    FIND_N_PRIMES = 20
    MAX_PRIME = int(1e4)
    SMALL_PRIME_LIST = ZUtils.find_primes(randint(3,MAX_PRIME), MAX_PRIME, cnt=FIND_N_PRIMES)

    def test_0init_ext(self):
        p = 1009
        ZField(p)

        ## Init given number as bignum < prime
        a = p/2
        a_bn = BigInt(a)
        zf  = ZFieldElExt(a_bn)
        self.assertTrue(zf.as_long() == a_bn.as_long())

        ## Init given number ZFieldElExt
        a = p/2
        a_bn = BigInt(a)
        zf    = ZFieldElExt(a_bn)
        zf2   = ZFieldElExt(zf)
        self.assertTrue(zf2.as_long() == a_bn.as_long())

        ## Init given number as bignum > prime
        a = p**2
        a_bn = BigInt(a)
        zf   =ZFieldElExt(a_bn)
        self.assertTrue(zf.as_long() == a % p)

    def test_1init_redc(self):
        p = 1009
        ZField(p)

        ## Init given number as bignum < prime
        a = p/2
        a_bn = BigInt(a)
        zf  = ZFieldElRedc(a_bn)
        self.assertTrue(zf.as_long() == a_bn.as_long())

        ## Init given number ZFieldElExt
        a = p/2
        a_bn = BigInt(a)
        zf    = ZFieldElRedc(a_bn)
        zf2   = ZFieldElRedc(zf)
        self.assertTrue(zf2.as_long() == a_bn.as_long())

        ## Init given number as bignum > prime
        a = p**2
        a_bn = BigInt(a)
        zf   =ZFieldElRedc(a_bn)
        self.assertTrue(zf.as_long() == a % p)


    def test_2conv_small_prime(self):

        for prime in ZFieldElTest.SMALL_PRIME_LIST:
           ZField(prime)
           reduction_data = ZField.get_reduction_data()

           for i in xrange(ZFieldElTest.TEST_ITER):
              x_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
              xz_ext = ZFieldElExt(x_l)
              xz_redc = xz_ext.reduce()
              rz_ext = xz_redc.extend()

              self.assertTrue(xz_ext.as_long() == rz_ext.as_long())
              self.assertTrue(xz_redc.as_long() == ((xz_ext.as_long() << reduction_data['Rbitlen']) % prime ))

    def test_3conv_large_prime(self):

       prime = ZField.FIELD_DATA['BN128']['prime']
       ZField(prime, ZField.FIELD_DATA['BN128']['field'])
       reduction_data = ZField.get_reduction_data()

       for i in xrange(ZFieldElTest.TEST_ITER):
         x_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
         xz_ext = ZFieldElExt(x_l)
         xz_redc = xz_ext.reduce()
         rz_ext = xz_redc.extend()

         self.assertTrue(xz_ext.as_long() == rz_ext.as_long())
         self.assertTrue(xz_redc.as_long() == ((xz_ext.as_long() << reduction_data['Rbitlen']) % prime ))

    def test_4arithmetic_small_prime(self):

        for prime in ZFieldElTest.SMALL_PRIME_LIST:
          ZField(prime)

          for i in xrange(ZFieldElTest.TEST_ITER):
            # init operands
            x_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
            y_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
            e_l = randint(1, ZFieldElTest.INT_THR)
            
            x_z = ZFieldElExt(x_l)
            y_z = ZFieldElExt(y_l)
            e_z = ZFieldElExt(e_l)

            x_zr = x_z.reduce()
            y_zr = y_z.reduce()
            e_zr = e_z.reduce()

            self.assertTrue(x_z.as_long() < prime)
            self.assertTrue(y_z.as_long() < prime)
            self.assertTrue(e_z.as_long() < prime)

            # +
            r_l  = (x_l + y_l) % prime
            r_z  = x_z + y_z
            r_zr = x_zr + y_zr
            r2_z  = r_zr.extend()

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_z.as_long() == r2_z.as_long())
            self.assertTrue(x_z.as_long() < prime)
            self.assertTrue(y_z.as_long() < prime)
            self.assertTrue(e_z.as_long() < prime)

            # -
            r_l = (x_l - y_l) % prime
            r_z = x_z - y_z
            r_zr = x_zr - y_zr
            r2_z  = r_zr.extend()

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_z.as_long() == r2_z.as_long())
            self.assertTrue(x_z.as_long() < prime)
            self.assertTrue(y_z.as_long() < prime)
            self.assertTrue(e_z.as_long() < prime)

            # *
            r_l = (x_l * y_l) % prime
            r_z = x_z * y_z
            r_zr = x_zr * y_zr
            r2_z  = r_zr.extend()

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_z.as_long() == r2_z.as_long())
            self.assertTrue(x_z.as_long() < prime)
            self.assertTrue(y_z.as_long() < prime)
            self.assertTrue(e_z.as_long() < prime)

            # // TODO -> // not implemented for montgomery
            # TODO does % work for montgomery?
            if y_z > 0:
              r_z = x_z // y_z
              r_z2 = x_z % y_z

              self.assertTrue(x_z.as_long() == ((r_z * y_z + r_z2).as_long()))
              self.assertTrue(x_z.as_long() < prime)
              self.assertTrue(y_z.as_long() < prime)
              self.assertTrue(e_z.as_long() < prime)


            # pow
            r_l = pow(x_l,e_l, prime)
            r_z = x_z ** e_l
            r_zr = x_zr ** e_l
            r2_z  = r_zr.extend()

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_z.as_long() == r2_z.as_long())
            self.assertTrue(x_z.as_long() < prime)
            self.assertTrue(y_z.as_long() < prime)
            self.assertTrue(e_z.as_long() < prime)

            # += 
            r_l = x_l
            r_l = (r_l + y_l) % prime
            r_z = x_z
            r_z += y_z 
            r_zr = x_zr
            r_zr += y_zr 
            r2_z  = r_zr.extend()

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_z.as_long() == r2_z.as_long())
            self.assertTrue(x_z.as_long() < prime)
            self.assertTrue(y_z.as_long() < prime)
            self.assertTrue(e_z.as_long() < prime)

            # -= 
            r_l = x_l
            r_l = (r_l - y_l) % prime
            r_z = x_z 
            r_z -= y_z 
            r_zr = x_zr
            r_zr -= y_zr
            r2_z  = r_zr.extend()

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_z.as_long() == r2_z.as_long())
            self.assertTrue(x_z.as_long() < prime)
            self.assertTrue(y_z.as_long() < prime)
            self.assertTrue(e_z.as_long() < prime)

            # neg
            r_l = -x_l % prime
            r_z = -x_z
            r_zr = -x_zr 
            r2_z  = r_zr.extend()

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_z.as_long() == r2_z.as_long())
            self.assertTrue(x_z.as_long() < prime)
            self.assertTrue(y_z.as_long() < prime)
            self.assertTrue(e_z.as_long() < prime)

    def test_5arithmetic_large_prime(self):

       prime = ZField.FIELD_DATA['BN128']['prime']
       ZField(prime, ZField.FIELD_DATA['BN128']['field'])

       for i in xrange(ZFieldElTest.TEST_ITER):
          # init operands
          x_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
          y_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
          e_l = randint(1, ZFieldElTest.INT_THR)
            
          x_z = ZFieldElExt(x_l)
          y_z = ZFieldElExt(y_l)
          e_z = ZFieldElExt(e_l)

          x_zr = x_z.reduce()
          y_zr = y_z.reduce()
          e_zr = e_z.reduce()

          # +
          r_l  = (x_l + y_l) % prime
          r_z  = x_z + y_z
          r_zr = x_zr + y_zr
          r2_z  = r_zr.extend()

          self.assertTrue(r_z.as_long() == long(r_l))
          self.assertTrue(r_z.as_long() == r2_z.as_long())

          # -
          r_l = (x_l - y_l) % prime
          r_z = x_z - y_z
          r_zr = x_zr - y_zr
          r2_z  = r_zr.extend()

          self.assertTrue(r_z.as_long() == long(r_l))
          self.assertTrue(r_z.as_long() == r2_z.as_long())

          # *
          r_l = (x_l * y_l) % prime
          r_z = x_z * y_z
          r_zr = x_zr * y_zr
          r2_z  = r_zr.extend()

          self.assertTrue(r_z.as_long() == long(r_l))
          self.assertTrue(r_z.as_long() == r2_z.as_long())

          # // TODO -> // not implemented for montgomery
          # TODO does % work for montgomery
          if y_z > 0:
            r_z = x_z // y_z
            r_z2 = x_z % y_z

            self.assertTrue(x_z.as_long() == ((r_z * y_z + r_z2).as_long()))

          # pow
          r_l = pow(x_l,e_l, prime)
          r_z = x_z ** e_l
          r_zr = x_zr ** e_l
          r2_z  = r_zr.extend()

          self.assertTrue(r_z.as_long() == long(r_l))
          self.assertTrue(r_z.as_long() == r2_z.as_long())

          # += 
          r_l = x_l
          r_l = (r_l + y_l) % prime
          r_z = x_z 
          r_z += y_z 
          r_zr = x_zr
          r_zr += y_zr 
          r2_z  = r_zr.extend()

          self.assertTrue(r_z.as_long() == long(r_l))
          self.assertTrue(r_z.as_long() == r2_z.as_long())

          # -= 
          r_l = x_l
          r_l = (r_l - y_l) % prime
          r_z = x_z 
          r_z -= y_z 
          r_zr = x_zr
          r_zr -= y_zr
          r2_z  = r_zr.extend()

          self.assertTrue(r_z.as_long() == long(r_l))
          self.assertTrue(r_z.as_long() == r2_z.as_long())

          # neg
          r_l = -x_l % prime
          r_z = -x_z
          r_zr = -x_zr 
          r2_z  = r_zr.extend()

          self.assertTrue(r_z.as_long() == long(r_l))
          self.assertTrue(r_z.as_long() == r2_z.as_long())

    def test_6bitwise_small_prime(self):

        for prime in ZFieldElTest.SMALL_PRIME_LIST:
          ZField(prime)

          for i in xrange(ZFieldElTest.TEST_ITER):
            # init operands
            x_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
            y_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
            e_l = randint(0, int(math.log(prime,2)))
            
            x_z = ZFieldElExt(x_l)
            y_z = ZFieldElExt(y_l)
            e_z = ZFieldElExt(e_l)

            x_zr = x_z.reduce()
            y_zr = y_z.reduce()
            e_zr = e_z.reduce()

            # <<
            r_l  = (x_l % prime) << e_l
            r_z  = x_z << e_l
            r_zr = x_zr << e_l

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_zr.as_long() == (x_zr.as_long() << e_l))

            # >>
            r_l  = (x_l % prime) >> e_l
            r_z  = x_z >> e_l
            r_zr = x_zr >> e_l

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_zr.as_long() == (x_zr.as_long() >> e_l))

            # &
            r_l  = (x_l % prime) & e_l
            r_z  = x_z & e_l
            r_zr = x_zr & e_l

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_zr.as_long() == (x_zr.as_long() & e_l))

            # |
            r_l  = (x_l % prime) | e_l
            r_z  = x_z | e_l
            r_zr = x_zr | e_l

            self.assertTrue(r_z.as_long() == long(r_l))
            self.assertTrue(r_zr.as_long() == (x_zr.as_long() | e_l))

    def test_7bitwise_large_prime(self):

       prime = ZField.FIELD_DATA['BN128']['prime']
       ZField(prime, ZField.FIELD_DATA['BN128']['field'])

       for i in xrange(ZFieldElTest.TEST_ITER):
         # init operands
         x_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
         y_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
         e_l = randint(0, int(math.log(prime,2)))
            
         x_z = ZFieldElExt(x_l)
         y_z = ZFieldElExt(y_l)
         e_z = ZFieldElExt(e_l)

         x_zr = x_z.reduce()
         y_zr = y_z.reduce()
         e_zr = e_z.reduce()

         # <<
         r_l  = (x_l % prime) << e_l
         r_z  = x_z << e_l
         r_zr = x_zr << e_l

         self.assertTrue(r_z.as_long() == long(r_l))
         self.assertTrue(r_zr.as_long() == (x_zr.as_long() << e_l))

         # >>
         r_l  = (x_l % prime) >> e_l
         r_z  = x_z >> e_l
         r_zr = x_zr >> e_l

         self.assertTrue(r_z.as_long() == long(r_l))
         self.assertTrue(r_zr.as_long() == (x_zr.as_long() >> e_l))

         # &
         r_l  = (x_l % prime) & e_l
         r_z  = x_z & e_l
         r_zr = x_zr & e_l

         self.assertTrue(r_z.as_long() == long(r_l))
         self.assertTrue(r_zr.as_long() == (x_zr.as_long() & e_l))

         # |
         r_l  = (x_l % prime) | e_l
         r_z  = x_z | e_l
         r_zr = x_zr | e_l

         self.assertTrue(r_z.as_long() == long(r_l))
         self.assertTrue(r_zr.as_long() == (x_zr.as_long() | e_l))

    def test_8comparison_small_prime(self):

        for prime in ZFieldElTest.SMALL_PRIME_LIST:
          ZField(prime)

          for i in xrange(ZFieldElTest.TEST_ITER):
            # init operands
            x_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
            y_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
            e_l = randint(0, int(math.log(prime,2)))
            
            x_z = ZFieldElExt(x_l)
            y_z = ZFieldElExt(y_l)
            e_z = ZFieldElExt(e_l)

            x_zr = x_z.reduce()
            y_zr = y_z.reduce()
            e_zr = e_z.reduce()

            # <
            r_l  = ((x_l % prime) < (y_l % prime))
            r_z  = x_z  < y_z
            r_zr = x_zr < y_zr
            r_l2 = x_zr.as_long() < y_zr.as_long()

            self.assertTrue(r_z == r_l)
            self.assertTrue(r_l2== r_zr)

            # <=
            r_l  = ((x_l % prime) <= (y_l % prime))
            r_z  = x_z  <= y_z
            r_zr = x_zr <= y_zr
            r_l2 = x_zr.as_long() <= y_zr.as_long()

            self.assertTrue(r_z == r_l)
            self.assertTrue(r_l2== r_zr)

            # >
            r_l  = ((x_l % prime) > (y_l % prime))
            r_z  = x_z  > y_z
            r_zr = x_zr > y_zr
            r_l2 = x_zr.as_long() > y_zr.as_long()

            self.assertTrue(r_z == r_l)
            self.assertTrue(r_l2== r_zr)

            # >=
            r_l  = ((x_l % prime) >= (y_l % prime))
            r_z  = x_z  >= y_z
            r_zr = x_zr >= y_zr
            r_l2 = x_zr.as_long() >= y_zr.as_long()

            self.assertTrue(r_z == r_l)
            self.assertTrue(r_l2== r_zr)

            # ==
            r_l  = x_l  == (x_l+2)
            r_z  = x_z  == (x_z+2)
            r_zr = x_zr == (x_zr+2)

            self.assertTrue(r_z == r_l)
            self.assertTrue(r_z == r_zr)
            self.assertTrue(r_z == False)

            # ==
            r_l  = x_l  == x_l
            r_z  = x_z  == x_z
            r_zr = x_zr == x_zr

            self.assertTrue(r_z == r_l)
            self.assertTrue(r_z == r_zr)
            self.assertTrue(r_z == True)

            # !=
            r_l  = x_l  != (x_l+2)
            r_z  = x_z  != (x_z+2)
            r_zr = x_zr != (x_zr+2)

            self.assertTrue(r_z == r_l)
            self.assertTrue(r_z == r_zr)
            self.assertTrue(r_z == True)

            # !=
            r_l  = x_l  != x_l
            r_z  = x_z  != x_z
            r_zr = x_zr != x_zr

            self.assertTrue(r_z == r_l)
            self.assertTrue(r_z == r_zr)
            self.assertTrue(r_z == False)

    def test_9comparison_large_prime(self):
       prime = ZField.FIELD_DATA['BN128']['prime']
       ZField(prime, ZField.FIELD_DATA['BN128']['field'])

       for i in xrange(ZFieldElTest.TEST_ITER):
           # init operands
           x_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
           y_l = randint(-ZFieldElTest.LONG_THR, ZFieldElTest.LONG_THR)
           e_l = randint(0, int(math.log(prime, 2)))

           x_z = ZFieldElExt(x_l)
           y_z = ZFieldElExt(y_l)
           e_z = ZFieldElExt(e_l)

           x_zr = x_z.reduce()
           y_zr = y_z.reduce()
           e_zr = e_z.reduce()

           # <
           r_l = ((x_l % prime) < (y_l % prime))
           r_z = x_z < y_z
           r_zr = x_zr < y_zr
           r_l2 = x_zr.as_long() < y_zr.as_long()

           self.assertTrue(r_z == r_l)
           self.assertTrue(r_l2 == r_zr)

           # <=
           r_l = ((x_l % prime) <= (y_l % prime))
           r_z = x_z <= y_z
           r_zr = x_zr <= y_zr
           r_l2 = x_zr.as_long() <= y_zr.as_long()

           self.assertTrue(r_z == r_l)
           self.assertTrue(r_l2 == r_zr)

           # >
           r_l = ((x_l % prime) > (y_l % prime))
           r_z = x_z > y_z
           r_zr = x_zr > y_zr
           r_l2 = x_zr.as_long() > y_zr.as_long()

           self.assertTrue(r_z == r_l)
           self.assertTrue(r_l2 == r_zr)

           # >=
           r_l = ((x_l % prime) >= (y_l % prime))
           r_z = x_z >= y_z
           r_zr = x_zr >= y_zr
           r_l2 = x_zr.as_long() >= y_zr.as_long()

           self.assertTrue(r_z == r_l)
           self.assertTrue(r_l2 == r_zr)

           # ==
           r_l = x_l == (x_l + 2)
           r_z = x_z == (x_z + 2)
           r_zr = x_zr == (x_zr + 2)

           self.assertTrue(r_z == r_l)
           self.assertTrue(r_z == r_zr)
           self.assertTrue(r_z == False)

           # ==
           r_l = x_l == x_l
           r_z = x_z == x_z
           r_zr = x_zr == x_zr

           self.assertTrue(r_z == r_l)
           self.assertTrue(r_z == r_zr)
           self.assertTrue(r_z == True)

           # !=
           r_l = x_l != (x_l + 2)
           r_z = x_z != (x_z + 2)
           r_zr = x_zr != (x_zr + 2)

           self.assertTrue(r_z == r_l)
           self.assertTrue(r_z == r_zr)
           self.assertTrue(r_z == True)

           # !=
           r_l = x_l != x_l
           r_z = x_z != x_z
           r_zr = x_zr != x_zr

           self.assertTrue(r_z == r_l)
           self.assertTrue(r_z == r_zr)
           self.assertTrue(r_z == False)


if __name__ == "__main__":
    unittest.main()

