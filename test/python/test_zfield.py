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
import sys
import unittest
from random import randint

import numpy as np

sys.path.append('../../src/python')

from zfield import *
from zutils import *


class ZFieldTest(unittest.TestCase):
    TEST_ITER = 1000
    FIND_N_PRIMES = 20
    NROOTS_THR = 15
    MAX_PRIME = int(1e4)

    SMALL_PRIME_LIST = ZUtils.find_primes(randint(3, MAX_PRIME), MAX_PRIME, cnt=FIND_N_PRIMES)

    def test0_init(self):

        ## Check ZField is not initialized 
        self.assertTrue(ZField.is_init() == False)
        self.assertTrue(ZField.get_extended_p() == None)
        self.assertTrue(ZField.get_reduced_p() == None)
        self.assertTrue(len(ZField.get_roots()[0]) == 0)
        self.assertTrue(len(ZField.get_roots()[1]) == 0)
        self.assertTrue(len(ZField.get_factors()['factors']) == 0)
        self.assertTrue(len(ZField.get_factors()['exponents']) == 0)

        ## Init given number as dec string and check all values are init (results not important)
        a_str = "1009"
        ZField(a_str, factor_data = ZUtils.CURVE_DATA['P1009']['curve'])

        self.assertTrue(ZField.get_extended_p().as_long() == long(a_str, 10))
        self.assertTrue(ZField.is_init() == True)
        self.assertTrue(isinstance(ZField.get_extended_p(), BigInt))
        self.assertTrue(isinstance(ZField.get_reduced_p(), BigInt))
        self.assertTrue(len(ZField.get_factors()['factors']) > 0)
        self.assertTrue(len(ZField.get_factors()['exponents']) > 0)

        ## Init given number as hex string
        a_str = "1009"
        ZField(hex(long(a_str)), ZUtils.CURVE_DATA['P1009']['curve'])

        self.assertTrue(ZField.get_extended_p().as_long() == long(a_str, 10))

        # Check reduction is correctly initialized 
        # R * Rp - P * Pp = 1
        a_str = "1009"
        ZField(hex(long(a_str)), ZUtils.CURVE_DATA['P1009']['curve'])

        r_data = ZField.get_reduction_data()

        rrp_nnp = r_data['R'] * r_data['Rp'] - r_data['Pp'] * ZField.get_extended_p().as_long()

        self.assertTrue(isinstance(ZField.get_reduced_p(), BigInt))
        self.assertTrue(rrp_nnp == 1)

        # Check reduction is correctly initialized  - BN128
        # R * Rp - P * Pp = 1
        a_str = ZUtils.CURVE_DATA['BN128']['prime']
        ZField(hex(long(a_str)), ZUtils.CURVE_DATA['BN128']['curve'])

        r_data = ZField.get_reduction_data()

        rrp_nnp = r_data['R'] * r_data['Rp'] - r_data['Pp'] * ZField.get_extended_p().as_long()

        self.assertTrue(isinstance(ZField.get_reduced_p(), BigInt))
        self.assertTrue(rrp_nnp == 1)

        # Check factorization - P1009 with no input data
        a_str = "1009"
        ZField(a_str)

        factor_data = ZField.get_factors()
        self.assertTrue(factor_data['factors'] == ZUtils.CURVE_DATA['P1009']['factor_data']['factors'])
        self.assertTrue(factor_data['exponents'] == ZUtils.CURVE_DATA['P1009']['factor_data']['exponents'])

        # Check factorization - P1009 with input data
        a_str = "1009"
        ZField(a_str, ZUtils.CURVE_DATA['P1009']['factor_data'])

        factor_data = ZField.get_factors()
        self.assertTrue(factor_data['factors'] == ZUtils.CURVE_DATA['P1009']['factor_data']['factors'])
        self.assertTrue(factor_data['exponents'] == ZUtils.CURVE_DATA['P1009']['factor_data']['exponents'])

        # Check factorization - BN128 with input data
        a_str = ZUtils.CURVE_DATA['BN128']['prime']
        ZField(a_str, ZUtils.CURVE_DATA['BN128']['factor_data'])

        factor_data = ZField.get_factors()
        self.assertTrue(factor_data['factors'] == ZUtils.CURVE_DATA['BN128']['factor_data']['factors'])
        self.assertTrue(factor_data['exponents'] == ZUtils.CURVE_DATA['BN128']['factor_data']['exponents'])

    def test1_reduction_and_factorization(self):

        for prime in ZFieldTest.SMALL_PRIME_LIST:
            # Check reduction is correctly initialized for multiple primes
            ZField(prime)

            r_data = ZField.get_reduction_data()
            f_data = ZField.get_factors()

            rrp_nnp = r_data['R'] * r_data['Rp'] - r_data['Pp'] * ZField.get_extended_p().as_long()

            # R * Rp - P * Pp = 1
            self.assertTrue(rrp_nnp == 1)

            self.assertTrue(len(f_data['factors']) == len(f_data['exponents']))

            factor_result = np.prod([f ** e for f, e in zip(f_data['factors'], f_data['exponents'])])

            self.assertTrue(factor_result == ZField.get_extended_p().as_long() - 1)

    def test2_generator_small_primes(self):
        # For a list of primes, check that generator can generate all the field

        for prime in ZFieldTest.SMALL_PRIME_LIST:
            ZField(prime)
            gen = ZField.find_generator().as_long()

            el = []

            for i in xrange(prime-1):
                el.append(pow(gen, i, prime))

            self.assertTrue(len(set(el)) == prime - 1)
            self.assertTrue(np.sum(el) == prime * (prime - 1) / 2)

    def test3_generator_large_primes(self):
        # For a list of primes, check that generator can generate all the field

        p = ZUtils.CURVE_DATA['BN128']['prime']
        ZField(p, ZUtils.CURVE_DATA['BN128']['curve'])
        gen = ZField.find_generator().as_long()

        for iter in xrange(ZFieldTest.TEST_ITER):
            t = randint(1, p)
            el = pow(gen, t, p)

            self.assertTrue(el != 1)

    def test4_roots_small_primes(self):
        # Check correct number of different roots are generated

        for prime in ZFieldTest.SMALL_PRIME_LIST:
            ZField(prime)
            f_data = ZField.get_factors()
            idx = randint(0, len(f_data['factors']) - 1)
            nroots = f_data['factors'][idx] ** randint(1, f_data['exponents'][idx])
            root, inv_root = ZField.find_roots(nroots)

            root_1 = root[1] * root[-1]
            root_l = [r.as_long() for r in root]
            inv_root_l = [r.as_long() for r in inv_root]

            self.assertTrue(len(set(root_l)) == nroots)
            self.assertTrue(len(set(inv_root_l)) == nroots)
            self.assertTrue(root_1.as_long() == 1)
            self.assertTrue(len(ZField.get_roots()[0]) == nroots)
            self.assertTrue(len(ZField.get_roots()[1]) == nroots)

            root_rdc, inv_root_rdc = ZField.find_roots(nroots, rformat_ext=False)

            r_s = set([x.as_long() for x in root])
            rrdc_s = set([x.extend().as_long() for x in root_rdc])
            rinv_s = set([x.as_long() for x in inv_root])
            rinvdc_s = set([x.extend().as_long() for x in inv_root_rdc])

            self.assertTrue(r_s == rrdc_s)
            self.assertTrue(rinv_s == rinvdc_s)

    def test5_roots_large_prime(self):
        p = ZUtils.CURVE_DATA['BN128']['prime']
        ZField(p, ZUtils.CURVE_DATA['BN128']['curve'])

        for i in xrange(ZFieldTest.NROOTS_THR - 1):
            nroots = 2 ** (i + 1)
            root, inv_root = ZField.find_roots(nroots)

            root_1 = root[1] * root[-1]
            root_l = [r.as_long() for r in root]
            inv_root_l = [r.as_long() for r in inv_root]

            self.assertTrue(len(set(root_l)) == nroots)
            self.assertTrue(len(set(inv_root_l)) == nroots)
            self.assertTrue(root_1.as_long() == 1)

            root_rdc, inv_root_rdc = ZField.find_roots(nroots, rformat_ext=False)

            r_s = set([x.as_long() for x in root])
            rrdc_s = set([x.extend().as_long() for x in root_rdc])
            rinv_s = set([x.as_long() for x in inv_root])
            rinvdc_s = set([x.extend().as_long() for x in inv_root_rdc])

            self.assertTrue(r_s == rrdc_s)
            self.assertTrue(rinv_s == rinvdc_s)
    def test6_inv_small_primes(self):
        for prime in ZFieldTest.SMALL_PRIME_LIST:
            ZField(prime)

            for i in xrange(ZFieldTest.TEST_ITER):
                x = ZFieldElExt(randint(1, prime - 1))
                x_inv = ZField.inv(x)
                r = x * x_inv

                self.assertTrue(r.as_long() == 1)

    def test7_inv_large_prime(self):
        prime = ZUtils.CURVE_DATA['BN128']['prime']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

        for i in xrange(ZFieldTest.TEST_ITER):
            x = ZFieldElExt(randint(1, prime - 1))
            x_inv = ZField.inv(x)
            r = x * x_inv

            self.assertTrue(r.as_long() == 1)


if __name__ == "__main__":
    unittest.main()
