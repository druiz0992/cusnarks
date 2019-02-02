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
// Date       : 30/01/2019
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
import sys
import unittest

sys.path.append('../../src/python')

from zfield import *
from z2field_element import *
from zutils import *


class Z2FieldElTest(unittest.TestCase):
    TEST_ITER = 1000

    def test_0init_ext(self):
        prime = ZUtils.CURVE_DATA['BN128']['prime_r']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

        ## Init 
        el2 = [ZFieldElExt(randint(1, prime - 1)), ZFieldElExt(randint(1, prime - 1))]
        el2_rdc = [el2[0].reduce(), el2[1].reduce()]
        el2_ext = [el2_rdc[0].extend(), el2_rdc[1].extend()]

        z2el_rdc = Z2FieldEl(el2_rdc)
        z2el_ext = Z2FieldEl(el2_ext)

        self.assertTrue(z2el_rdc == z2el_ext.reduce())
        self.assertTrue(z2el_rdc.extend() == z2el_ext)

        # Object
        el2 = [ZFieldElExt(randint(1, prime - 1)), ZFieldElExt(randint(1, prime - 1))]
        el2_rdc = [el2[0].reduce(), el2[1].reduce()]
        el2_ext = [el2_rdc[0].extend(), el2_rdc[1].extend()]

        z2el_rdc = Z2FieldEl(el2_rdc)
        z2el_ext = Z2FieldEl(el2_ext)

        r2el_rdc = Z2FieldEl(z2el_rdc)
        r2el_ext = Z2FieldEl(z2el_ext)

        self.assertTrue(z2el_rdc == r2el_rdc)
        self.assertTrue(z2el_ext == r2el_ext)

        ## none
        z2el = Z2FieldEl(None)
        self.assertTrue(z2el.as_list() == Z2FieldEl.zero[ZUtils.DEFAULT_IN_REP_FORMAT])

    def test_1reduce_extend(self):

        prime = ZUtils.CURVE_DATA['BN128']['prime_r']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

        for i in xrange(Z2FieldElTest.TEST_ITER):
            el2 = [ZFieldElExt(randint(1, prime - 1)), ZFieldElExt(randint(1, prime - 1))]
            el2_rdc = [el2[0].reduce(), el2[1].reduce()]
            el2_ext = [el2_rdc[0].extend(), el2_rdc[1].extend()]

            z2el_rdc = Z2FieldEl(el2_rdc)
            z2el_ext = Z2FieldEl(el2_ext)

            self.assertTrue(z2el_rdc == z2el_ext.reduce())
            self.assertTrue(z2el_rdc.extend() == z2el_ext)
            self.assertTrue(isinstance(z2el_rdc.P[0], ZFieldElRedc) and isinstance(z2el_rdc.P[1], ZFieldElRedc))
            self.assertTrue(isinstance(z2el_ext.P[0], ZFieldElExt) and isinstance(z2el_ext.P[1], ZFieldElExt))

    def test_2arithmetic(self):

        prime = ZUtils.CURVE_DATA['BN128']['prime_r']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

        for i in xrange(Z2FieldElTest.TEST_ITER):
            el2 = [ZFieldElExt(randint(1, prime - 1)), ZFieldElExt(randint(1, prime - 1))]
            el2_rdc = [el2[0].reduce(), el2[1].reduce()]
            el2_ext = [el2_rdc[0].extend(), el2_rdc[1].extend()]

            z2el1_rdc = Z2FieldEl(el2_rdc)
            z2el1_ext = Z2FieldEl(el2_ext)

            el2 = [ZFieldElExt(randint(1, prime - 1)), ZFieldElExt(randint(1, prime - 1))]
            el2_rdc = [el2[0].reduce(), el2[1].reduce()]
            el2_ext = [el2_rdc[0].extend(), el2_rdc[1].extend()]

            z2el2_rdc = Z2FieldEl(el2_rdc)
            z2el2_ext = Z2FieldEl(el2_ext)

            scalar = ZFieldElExt(randint(1, prime - 1))

            ####
            # +
            z2_rext = z2el1_ext + z2el2_ext
            z2_rrdc = z2el1_rdc + z2el2_rdc

            self.assertTrue(z2_rext == z2_rrdc.extend())
            self.assertTrue(z2_rext.reduce() == z2_rrdc)
            self.assertTrue(z2_rext.P[0] == z2el1_ext.P[0] + z2el2_ext.P[0])
            self.assertTrue(z2_rext.P[1] == z2el1_ext.P[1] + z2el2_ext.P[1])

            # -
            z2_rext = z2el1_ext - z2el2_ext
            z2_rrdc = z2el1_rdc - z2el2_rdc

            self.assertTrue(z2_rext == z2_rrdc.extend())
            self.assertTrue(z2_rext.reduce() == z2_rrdc)
            self.assertTrue(z2_rext.P[0] == z2el1_ext.P[0] - z2el2_ext.P[0])
            self.assertTrue(z2_rext.P[1] == z2el1_ext.P[1] - z2el2_ext.P[1])

            # neg
            z2_rext = -z2el2_ext
            z2_rrdc = -z2el2_rdc

            self.assertTrue(z2_rext == z2_rrdc.extend())
            self.assertTrue(z2_rext.reduce() == z2_rrdc)
            self.assertTrue(z2_rext.P[0] == -z2el2_ext.P[0])
            self.assertTrue(z2_rext.P[1] == -z2el2_ext.P[1])

            # +=
            z2_rext = z2el1_ext
            z2_rext += z2el2_ext
            z2_rrdc = z2el1_rdc
            z2_rrdc += z2el2_rdc

            self.assertTrue(z2_rext == z2_rrdc.extend())
            self.assertTrue(z2_rext.reduce() == z2_rrdc)
            self.assertTrue(z2_rext.P[0] == z2el1_ext.P[0] + z2el2_ext.P[0])
            self.assertTrue(z2_rext.P[1] == z2el1_ext.P[1] + z2el2_ext.P[1])

            # *
            z2_rext = z2el1_ext * z2el2_ext
            z2_rrdc = z2el1_rdc * z2el2_rdc

            self.assertTrue(z2_rext == z2_rrdc.extend())
            self.assertTrue(z2_rext.reduce() == z2_rrdc)
            self.assertTrue(isinstance(z2_rrdc.P[0], ZFieldElRedc) and isinstance(z2_rrdc.P[1], ZFieldElRedc))
            self.assertTrue(isinstance(z2_rext.P[0], ZFieldElExt) and isinstance(z2_rext.P[1], ZFieldElExt))

            # scalar *
            z2_rext1 = scalar * z2el2_ext
            z2_rext2 = z2el2_ext * scalar

            self.assertTrue(z2_rext1 == z2_rext2)
            self.assertTrue(isinstance(z2_rext1.P[0], ZFieldElExt) and isinstance(z2_rext1.P[1], ZFieldElExt))

            alpha_ext = randint(1, 100)
            r1_ext = z2el2_ext * alpha_ext
            r2_ext = np.sum([z2el2_ext]*alpha_ext)

            self.assertTrue(r1_ext == r2_ext)

if __name__ == "__main__":
    unittest.main()
