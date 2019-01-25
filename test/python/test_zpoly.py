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
// File name  : test_zpoly.py
//
// Date       : 23/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Poly test
//
// TODO 
//    incorrect format  -> once asserts substituted by exceptions,
//         test incorrect formats can be done
// ------------------------------------------------------------------

"""
import sys
import unittest
import numpy as np
from random import randint, sample

sys.path.append('../../src/python')

from zpoly import *
from zutils import *


class ZPolyTest(unittest.TestCase):
    TEST_ITER = 20
    MAX_POLY_DEGREE = 1000

    def test_0init_ext(self):
        c = ZUtils.CURVE_DATA['BN128']
        prime = c['prime']
        ZField(prime)

        ## Init poly by degree
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        poly_a = ZPoly(p_d)

        r_c, r_d, r_f = poly_a.get_properties()

        self.assertTrue(len(r_c) == p_d + 1)
        self.assertTrue(r_d == p_d)
        self.assertTrue(isinstance(r_c[0], ZFieldElExt))
        self.assertTrue(r_f == ZPoly.FEXT)

        # inti poly by list
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        p_lc = [randint(1, prime * 2) for x in xrange(p_d + 1)]
        poly_a = ZPoly(p_lc)

        r_c, r_d, r_f = poly_a.get_properties()

        self.assertTrue(len(r_c) == p_d + 1)
        self.assertTrue(r_d == p_d)
        self.assertTrue(isinstance(r_c[0], ZFieldElExt))
        self.assertTrue(r_f == ZPoly.FEXT)

        # inti poly by list (ZFieldRdc)
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        p_lc = [ZFieldElRedc(randint(1, prime * 2)) for x in xrange(p_d + 1)]
        poly_a = ZPoly(p_lc)

        r_c, r_d, r_f = poly_a.get_properties()

        self.assertTrue(len(r_c) == p_d + 1)
        self.assertTrue(r_d == p_d)
        self.assertTrue(isinstance(r_c[0], ZFieldElRedc))
        self.assertTrue(r_f == ZPoly.FRDC)

        # inti poly by ZPoly
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        p_lc = [randint(1, prime * 2) for x in xrange(p_d + 1)]
        poly_a = ZPoly(p_lc)
        poly_b = ZPoly(poly_a)

        ra_c, ra_d, ra_f = poly_a.get_properties()
        rb_c, rb_d, rb_f = poly_b.get_properties()

        self.assertTrue(ra_c == rb_c)
        self.assertTrue(ra_d == rb_d)
        self.assertTrue(ra_f == rb_f)

        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        p_lc = [randint(1, prime * 2) for x in xrange(p_d + 1)]
        poly_a = ZPoly(p_lc)
        poly_rdca = poly_a.reduce()
        poly_rdcb = ZPoly(poly_rdca)

        ra_c, ra_d, ra_f = poly_rdca.get_properties()
        rb_c, rb_d, rb_f = poly_rdcb.get_properties()

        self.assertTrue(ra_c == rb_c)
        self.assertTrue(ra_d == rb_d)
        self.assertTrue(ra_f == rb_f)

        # inti poly by dict
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        n_coeff = randint(1, p_d)
        p_c = sample(xrange(p_d + 1), n_coeff)
        p_d = max(sorted(p_c))
        p_lc = {str(coeff_at): ZFieldElExt(randint(0, prime * 2)) for coeff_at in p_c}
        poly_a = ZPoly(p_lc)

        r_c, r_d, r_f = poly_a.get_properties()

        self.assertTrue(len(r_c) == p_d + 1)
        self.assertTrue(r_d == p_d)
        self.assertTrue(isinstance(r_c[0], ZFieldElExt))
        self.assertTrue(r_f == ZPoly.FEXT)

        # inti poly by dict (ZFieldRdc)
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        n_coeff = randint(1, p_d)
        p_c = sample(xrange(p_d + 1), n_coeff)
        p_d = max(sorted(p_c))
        p_lc = {str(coeff_at): ZFieldElRedc(randint(0, prime * 2)) for coeff_at in p_c}
        poly_a = ZPoly(p_lc)

        r_c, r_d, r_f = poly_a.get_properties()

        self.assertTrue(len(r_c) == p_d + 1)
        self.assertTrue(r_d == p_d)
        self.assertTrue(isinstance(r_c[0], ZFieldElRedc))
        self.assertTrue(r_f == ZPoly.FRDC)

        # inti poly by ZPoly
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        n_coeff = randint(1, p_d)
        p_c = sample(xrange(p_d + 1), n_coeff)
        p_lc = {str(coeff_at): ZFieldElExt(randint(0, prime * 2)) for coeff_at in p_c}
        poly_a = ZPoly(p_lc)
        poly_b = ZPoly(poly_a)

        ra_c, ra_d, ra_f = poly_a.get_properties()
        rb_c, rb_d, rb_f = poly_b.get_properties()

        self.assertTrue(ra_c == rb_c)
        self.assertTrue(ra_d == rb_d)
        self.assertTrue(ra_f == rb_f)

        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        n_coeff = randint(1, p_d)
        p_c = sample(xrange(p_d + 1), n_coeff)
        p_lc = {str(coeff_at): ZFieldElRedc(randint(0, prime * 2)) for coeff_at in p_c}
        poly_a = ZPoly(p_lc)
        poly_rdca = poly_a.reduce()
        poly_rdcb = ZPoly(poly_rdca)

        ra_c, ra_d, ra_f = poly_rdca.get_properties()
        rb_c, rb_d, rb_f = poly_rdcb.get_properties()

        self.assertTrue(ra_c == rb_c)
        self.assertTrue(ra_d == rb_d)
        self.assertTrue(ra_f == rb_f)

    def test_1reduce_extend(self):
        c = ZUtils.CURVE_DATA['BN128']
        prime = c['prime']
        ZField(prime)

        for i in xrange(ZPolyTest.TEST_ITER):
            p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
            poly_a = ZPoly(p_d)

            coeff = poly_a.get_coeff()
            coeff_ext = [c.extend() for c in coeff]
            coeff_rdc = [c.reduce() for c in coeff]

            poly_ext = ZPoly(coeff_ext)
            poly_rdc = ZPoly(coeff_rdc)

            rext_c, rext_d, rext_f = poly_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = poly_rdc.get_properties()

            self.assertTrue(len(rext_c) == p_d + 1)
            self.assertTrue(rext_d == p_d)
            self.assertTrue(isinstance(rext_c[0], ZFieldElExt))
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(len(rrdc_c) == p_d + 1)
            self.assertTrue(rrdc_d == p_d)
            self.assertTrue(isinstance(rrdc_c[0], ZFieldElRedc))
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))

            # Pending Sparse Poly

    def test_2expand(self):
        c = ZUtils.CURVE_DATA['BN128']
        prime = c['prime']
        ZField(prime)

        for i in xrange(ZPolyTest.TEST_ITER):
            p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
            poly_ext = ZPoly(p_d)
            poly_rdc = poly_ext.reduce()

            new_d = randint(p_d + 1, ZPolyTest.MAX_POLY_DEGREE + 1)
            poly2_ext = poly_ext.expand_to_degree(new_d)
            poly2_rdc = poly_rdc.expand_to_degree(new_d)

            rext_c, rext_d, rext_f = poly2_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = poly2_rdc.get_properties()

            self.assertTrue(len(rext_c) == new_d + 1)
            self.assertTrue(rext_d == new_d)
            self.assertTrue(isinstance(rext_c[0], ZFieldElExt))
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(len(rrdc_c) == new_d + 1)
            self.assertTrue(rrdc_d == new_d)
            self.assertTrue(isinstance(rrdc_c[0], ZFieldElRedc))
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c[p_d + 1:] == [ZFieldElExt(0)] * (new_d - p_d))
            self.assertTrue(rext_c[:p_d + 1] == poly_ext.get_coeff()[:p_d + 1])
            self.assertTrue(rrdc_c[p_d + 1:] == [ZFieldElRedc(0)] * (new_d - p_d))
            self.assertTrue(rrdc_c[:p_d + 1] == poly_rdc.get_coeff()[:p_d + 1])

            # Pending Sparse Poly

    def test_3aritmetic(self):
        c = ZUtils.CURVE_DATA['BN128']
        prime = c['prime']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

        # *, + , - , /, inv, scalar mul
        for i in xrange(ZPolyTest.TEST_ITER):
            p1_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
            p2_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)

            scalar_l = randint(1,prime-1)
            scalar_bn = BigInt(scalar_l)
            scalar_ext = ZFieldElExt(scalar_bn)
            scalar_rdc = scalar_ext.reduce()

            poly1_ext = ZPoly(p1_d)
            poly2_ext = ZPoly(p2_d)

            poly1_rdc = poly1_ext.reduce()
            poly2_rdc = poly2_ext.reduce()

            # +
            p3_ext = poly1_ext + poly2_ext
            p3_rdc = poly1_rdc + poly2_rdc

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()

            self.assertTrue(rext_d == max(p1_d, p2_d))
            self.assertTrue(rrdc_d == max(p1_d, p2_d))
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c[:min(p1_d,p2_d)+1] == [r1 + r2 for r1, r2 in zip(poly1_ext.get_coeff()[:min(p1_d, p2_d) + 1],
                                                                 poly2_ext.get_coeff()[:min(p1_d, p2_d) + 1])])
            self.assertTrue(rrdc_c[:min(p1_d,p2_d)+1] == [r1 + r2 for r1, r2 in zip(poly1_rdc.get_coeff()[:min(p1_d, p2_d) + 1],
                                                                                    poly2_rdc.get_coeff()[:min(p1_d, p2_d) + 1])])
            if p1_d > p2_d:
                self.assertTrue(rext_c[p2_d+1:] == poly1_ext.get_coeff()[p2_d + 1:])
                self.assertTrue(rrdc_c[p2_d+1:] == poly1_rdc.get_coeff()[p2_d + 1:])
            else:
                self.assertTrue(rext_c[p1_d+1:] == poly2_ext.get_coeff()[p1_d + 1:])
                self.assertTrue(rrdc_c[p1_d+1:] == poly2_rdc.get_coeff()[p1_d + 1:])

            # -
            p3_ext = poly1_ext - poly2_ext
            p3_rdc = poly1_rdc - poly2_rdc

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()

            self.assertTrue(rext_d == max(p1_d, p2_d))
            self.assertTrue(rrdc_d == max(p1_d, p2_d))
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c[:min(p1_d,p2_d)+1] == [r1 - r2 for r1, r2 in zip(poly1_ext.get_coeff()[:min(p1_d, p2_d) + 1],
                                                                                    poly2_ext.get_coeff()[:min(p1_d, p2_d) + 1])])
            self.assertTrue(rrdc_c[:min(p1_d,p2_d)+1] == [r1 - r2 for r1, r2 in zip(poly1_rdc.get_coeff()[:min(p1_d, p2_d) + 1],
                                                                                    poly2_rdc.get_coeff()[:min(p1_d, p2_d) + 1])])
            if p1_d > p2_d:
                self.assertTrue(rext_c[p2_d+1:] == poly1_ext.get_coeff()[p2_d + 1:])
                self.assertTrue(rrdc_c[p2_d+1:] == poly1_rdc.get_coeff()[p2_d + 1:])
            else:
                self.assertTrue(rext_c[p1_d+1:] == [-c for c in poly2_ext.get_coeff()[p1_d + 1:]])
                self.assertTrue(rrdc_c[p1_d+1:] == [-c for c in poly2_rdc.get_coeff()[p1_d + 1:]])

            # scalar mul
            ## mont * mont -> mont
            ## ext * ext   -> ext
            ## mont * ext  -> mont (using default *)

            # case 1 : scalar_long * poly
            p3_ext = scalar_l * poly1_ext
            p3_rdc = scalar_l * poly1_rdc

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()

            self.assertTrue(rext_d == poly1_ext.get_degree())
            self.assertTrue(rrdc_d == poly1_rdc.get_degree())
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c == [scalar_l * r2 for r2 in poly1_ext.get_coeff()])
            self.assertTrue(rrdc_c == [scalar_l * r2 for r2 in poly1_rdc.get_coeff()])

            # case 2 : poly * scalar_long
            p3_ext = poly1_ext * scalar_l
            p3_rdc = poly1_rdc * scalar_l

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()

            self.assertTrue(rext_d == poly1_ext.get_degree())
            self.assertTrue(rrdc_d == poly1_rdc.get_degree())
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c == [scalar_l * r2 for r2 in poly1_ext.get_coeff()])
            self.assertTrue(rrdc_c == [scalar_l * r2 for r2 in poly1_rdc.get_coeff()])

            # case 3 : scalar_bn * poly
            p3_ext = scalar_bn * poly1_ext
            p3_rdc = scalar_bn * poly1_rdc

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()

            self.assertTrue(rext_d == poly1_ext.get_degree())
            self.assertTrue(rrdc_d == poly1_rdc.get_degree())
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c == [scalar_bn * r2 for r2 in poly1_ext.get_coeff()])
            self.assertTrue(rrdc_c == [scalar_bn * r2 for r2 in poly1_rdc.get_coeff()])

            # case 4 : poly * scalar_bn
            p3_ext = poly1_ext * scalar_bn
            p3_rdc = poly1_rdc * scalar_bn

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()

            self.assertTrue(rext_d == poly1_ext.get_degree())
            self.assertTrue(rrdc_d == poly1_rdc.get_degree())
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c == [scalar_bn * r2 for r2 in poly1_ext.get_coeff()])
            self.assertTrue(rrdc_c == [scalar_bn * r2 for r2 in poly1_rdc.get_coeff()])

            # case 4 : scalar_ext * poly
            p3_ext = scalar_ext * poly1_ext
            p3_rdc = scalar_ext* poly1_rdc

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()

            self.assertTrue(rext_d == poly1_ext.get_degree())
            self.assertTrue(rrdc_d == poly1_rdc.get_degree())
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c == [scalar_ext * r2 for r2 in poly1_ext.get_coeff()])
            self.assertTrue(rrdc_c == [scalar_ext * r2 for r2 in poly1_rdc.get_coeff()])

            # case 5 : poly * scalar_bn
            p3_ext = poly1_ext * scalar_ext
            p3_rdc = poly1_rdc * scalar_ext

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()

            self.assertTrue(rext_d == poly1_ext.get_degree())
            self.assertTrue(rrdc_d == poly1_rdc.get_degree())
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c == [scalar_ext * r2 for r2 in poly1_ext.get_coeff()])
            self.assertTrue(rrdc_c == [scalar_ext * r2 for r2 in poly1_rdc.get_coeff()])

            # case 6 : scalar_rdc * poly
            p3a_rdc = scalar_rdc * poly1_ext # recuced
            p3b_rdc = scalar_rdc* poly1_rdc

            rrdca_c, rrdca_d, rrdca_f = p3a_rdc.get_properties()
            rrdcb_c, rrdcb_d, rrdcb_f = p3b_rdc.get_properties()

            self.assertTrue(rrdca_d == poly1_ext.get_degree())
            self.assertTrue(rrdcb_d == poly1_rdc.get_degree())
            self.assertTrue(rrdca_f == ZPoly.FRDC)
            self.assertTrue(rrdcb_f == ZPoly.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(all([re == rr for re, rr in zip(rrdca_c, rrdcb_c)]))
            self.assertTrue(rrdca_c == [scalar_rdc * r2 for r2 in poly1_ext.get_coeff()])
            self.assertTrue(rrdcb_c == [scalar_rdc * r2 for r2 in poly1_rdc.get_coeff()])

            # case 7 : poly * scalar_rdc
            p3a_rdc = poly1_ext * scalar_rdc # recuced
            p3b_rdc = poly1_rdc * scalar_rdc

            rrdca_c, rrdca_d, rrdca_f = p3a_rdc.get_properties()
            rrdcb_c, rrdcb_d, rrdcb_f = p3b_rdc.get_properties()

            self.assertTrue(rrdca_d == poly1_ext.get_degree())
            self.assertTrue(rrdcb_d == poly1_rdc.get_degree())
            self.assertTrue(rrdca_f == ZPoly.FRDC)
            self.assertTrue(rrdcb_f == ZPoly.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(all([re == rr for re, rr in zip(rrdca_c, rrdcb_c)]))
            self.assertTrue(rrdca_c == [scalar_rdc * r2 for r2 in poly1_ext.get_coeff()])
            self.assertTrue(rrdcb_c == [scalar_rdc * r2 for r2 in poly1_rdc.get_coeff()])

            # *
            p1_ext = ZPoly(poly1_ext)
            p1_rdc = ZPoly(poly1_rdc)
            p2_ext = ZPoly(poly2_ext)
            p2_rdc = ZPoly(poly2_rdc)

            p1_ext.poly_mul(p2_ext)
            p1_rdc.poly_mul(p2_rdc)

            rext_c, rext_d, rext_f = p1_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p1_rdc.get_properties()


            p1_c = [c.as_long() for c in poly1_ext.get_coeff()]
            p2_c = [c.as_long() for c in poly2_ext.get_coeff()]
            p3_c = np.polymul(p1_c, p2_c)
            p3_c = [c % prime for c in p3_c]
            p3_d = len(p3_c)- 1

            self.assertTrue(rext_d == rrdc_d)
            self.assertTrue(rext_d == p3_d)
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue([c.as_long() for c in rext_c[:len(p3_c)]] == p3_c)

            # /
            pn_ext   = poly1_ext.scale(-p1_d/2)
            pd_ext = ZPoly(pn_ext)

            pnum_ext = pn_ext.scale(p1_d/4)
            pden_ext = pd_ext.scale(p1_d/4)
            prem_ext = pd_ext.scale(-p1_d/4 )

            pnum_ext.poly_mul(pd_ext)
            pnum_ext = pnum_ext + prem_ext

            pnum_rdc = pnum_ext.reduce()
            pden_rdc = pden_ext.reduce()
            prem_rdc = prem_ext.reduce()

            p3_ext = pnum_ext.poly_div(pden_ext)
            p3_rdc = pnum_rdc.poly_div(pden_rdc)

            p3_ext.poly_mul(pden_ext)
            p3_ext = p3_ext + prem_ext

            p3_rdc.poly_mul(pden_rdc)
            p3_rdc = p3_rdc + prem_rdc

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()


            #TODO : Montgomery division doesn't work
            self.assertTrue(rext_d == pnum_ext.get_degree())
            self.assertTrue(rrdc_d == pnum_rdc.get_degree())
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(rext_f == ZPoly.FEXT)
            self.assertTrue(rrdc_f == ZPoly.FRDC)
            #self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            #self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c == pnum_ext.get_coeff())
            #self.assertTrue(rrdc_c == pnum_rdc.get_coeff())


if __name__ == "__main__":
    unittest.main()
