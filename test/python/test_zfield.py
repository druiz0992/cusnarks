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
// File name  : test_zfield.py
//
// Date       : 15/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Zfield class test
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

from zfield import *


class ZFieldTest(unittest.TestCase):

    TEST_ITER = 1000
    NROOTS_THR = 15
   
    def find_primes(n):
       # Initialize a list
       primes = []
       for possiblePrime in range(2, n):
           
           # Assume number is prime until shown it is not. 
           isPrime = True
           for num in range(2, int(possiblePrime ** 0.5) + 1):
               if possiblePrime % num == 0:
                   isPrime = False
                   break
             
           if isPrime:
               primes.append(possiblePrime)
       return primes

    def test_init(self):

        ## Check ZField is not initialized 
        self.assertTrue(ZField.is_init() == False)
        self.assertTrue(ZField.get_extended_p() == None)
        self.assertTrue(ZField.get_reduced_p() == None)
        self.assertTrue(length(ZField.get_roots()[0]) == 0)
        self.assertTrue(length(ZField.get_roots()[1]) == 0)
        self.assertTrue(length(ZField.get_factors()['factors']) == 0)
        self.assertTrue(length(ZField.get_factors()['exponents']) == 0)

        ## Init given number as dec string and check all values are init (results not important)
        a_str = "1009"
        ZField(a_str, ZField.P1009_DATA['curve'])

        self.assertTrue(ZField.get_extended_p().as_long() == long(a_str,10))
        self.assertTrue(ZField.is_init() == True)
        self.assertTrue(isinstance(ZField.get_extended_p(), BigInt))
        self.assertTrue(isinstance(ZField.get_reduced_p(), BigInt))
        self.assertTrue(length(ZField.get_roots()[0]) > 0)
        self.assertTrue(length(ZField.get_roots()[1]) > 0)
        self.assertTrue(length(ZField.get_factors()['factors']) > 0)
        self.assertTrue(length(ZField.get_factors()['exponents']) > 0)

        ## Init given number as hex string
        a_str = "1009"
        ZField(hex(long(a_str)), ZField.P1009_DATA['curve'])

        self.assertTrue(ZField.get_extended_p().as_long() == long(a_str,10))

        # Check reduction is correctly initialized 
        # R * Rp - P * Pp = 1
        a_str = "1009"
        ZField(hex(long(a_str)), ZField.P1009_DATA['curve'])

        r_data = ZField.get_reduce_data()

        rrp_nnp = r['R'] * r['Rp'] - r['Pp'] * ZField.get_extended_p().as_long()

        self.assertTrue(isinstance(ZField.get_reduced_p(), BigInt))
        self.assertTrue(rrp_nnp == 1)

        # Check reduction is correctly initialized  - BN128
        # R * Rp - P * Pp = 1
        a_str = ZField.BN128_DATA['prime']
        ZField(hex(long(a_str)), ZField.BN128_DATA['curve'])

        r_data = ZField.get_reduce_data()

        rrp_nnp = r['R'] * r['Rp'] - r['Pp'] * ZField.get_extended_p().as_long()

        self.assertTrue(isinstance(ZField.get_reduced_p(), BigInt))
        self.assertTrue(rrp_nnp == 1)

        # Check factorization - P1009 with no input data
        a_str = "1009"
        ZField(a_str)

        factor_data = ZField.get_factors()
        self.assertTrue( factor_data['factors'] == ZField.P1009_DATA['factor_data']['factors'])
        self.assertTrue( factor_data['exponents'] == ZField.P1009_DATA['factor_data']['exponents'])

        # Check factorization - P1009 with input data
        a_str = "1009"
        ZField(a_str, ZField.P1009_DATA['factor_data'])

        factor_data = ZField.get_factors()
        self.assertTrue( factor_data['factors'] == ZField.P1009_DATA['factor_data']['factors'])
        self.assertTrue( factor_data['exponents'] == ZField.P1009_DATA['factor_data']['exponents'])

        # Check factorization - BN128 with input data
        a_str = ZField.BN128_DATA['prime']
        ZField(a_str, ZField.BN128_DATA['factor_data'])

        factor_data = ZField.get_factors()
        self.assertTrue( factor_data['factors'] == ZField.BN128_DATA['factor_data']['factors'])
        self.assertTrue( factor_data['exponents'] == ZField.BN128_DATA['factor_data']['exponents'])

    def test_reduction_and_factorization(self):
        prime_list = find_primes(ZField.PRIME_THR)

        for prime in prime_list:
           # Check reduction is correctly initialized for multiple primes
           ZField(prime)

           r_data = ZField.get_reduce_data()
           f_data = ZField.get_factors()

           rrp_nnp = r_data['R'] * r_data['Rp'] - r_data['Pp'] * ZField.get_extended_p().as_long()

           # R * Rp - P * Pp = 1
           self.assertTrue(rrp_nnp == 1)

           self.assertTrue(length(f_data['factors']) == length(f_data['exponents']))
           factor_result = 0

           for f,e in zip(f_data['factors'],f_data['exponents']):
               factor_result += f ** e
           self.assertTrue(factor_result == ZField.get_reduced_p().as_long() - 1)

    def test_generator_small_primes(self):
        # For a list of primes, check that generator can generate all the field
        prime_list = find_primes(ZField.PRIME_THR)

        for prime in prime_list:
           ZField(prime)
           gen = ZField.find_generator()

           el = []

           for i in xrange(prime):
              el.append(pow(gen,i,prime))

           self.assertTrue(set(el) == prime-1)

    def test_generator_large_primes(self):
        # For a list of primes, check that generator can generate all the field

        p = ZField.BN128_DATA['prime']
        ZField(p, ZField.BN128_DATA['curve'])
        gen = ZField.find_generator()

        for iter in xrange(ZFieldTest.TEST_ITER):
           t = randint(1,ZField.p)
           el = pow(gen, t, prime)

           self.assertTrue(el != 1)

    def test_roots_small_primes(self):
        # Check correct number of different roots are generated
        prime_list = find_primes(ZField.PRIME_THR)

        for prime in prime_list:
           ZField(prime)
           f_data = ZField.get_factors()
           idx = randint(0,length(f_data['factors'])-1)
           nroots = f_data['factors'][idx] ** randint(1,f_data['exponents'])
           root, inv_root = ZField.find_roots(nroots)

           root_1 = root[1] * root[-1]

           self.assertTrue(set(root) == nroots)
           self.assertTrue(set(inv_root) == nroots)
           self.assertTrue(root_1 in set(root))

    def test_roots_large_prime(self):
        p = ZField.BN128_DATA['prime']
        ZField(p, ZField.BN128_DATA['curve'])
        f_data = ZField.get_factors()

        for i in xrange(ZFieldTest.NROOTS_THR):
           nroots = 2 ** i
           root, inv_root = ZField.find_roots(nroots)

           root_1 = root[1] * root[-1]

           self.assertTrue(set(root) == nroots)
           self.assertTrue(set(inv_root) == nroots)
           self.assertTrue(root_1 in set(root))


if __name__ == "__main__":
    unittest.main()

