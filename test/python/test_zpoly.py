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
// TODO  incorrect format  -> once asserts substituted by exceptions,
//         test incorrect formats can be done
// TODO after arithmetic operation, we normalize poly. so we need to make
//    sure that when we compare operation to expected result, expected result
//    is also normalized
// TODO check norm operation after arithmetic operation
//
// -----------------------------------------------------------------

"""
import sys
import unittest
import numpy as np
from random import randint, sample

sys.path.append('../../src/python')

from zpoly import *
from zutils import *


class ZPolyTest(unittest.TestCase):
    TEST_ITER = 10
    MAX_POLY_DEGREE = 200
    SHIFT_SCALAR = 70
    MAX_FFT_NROWS = 3
    MAX_FFT_NROWS_W = 3
    MAX_FFT_NCOLS = 3
    MAX_FFT_NCOLS_W = 3
    FFT_PARALLEL_N = 4

    def test_0init_ext(self):
        c = ZUtils.CURVE_DATA['BN128']
        prime = c['prime']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

        ## Init poly by degree
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        poly_a = ZPoly(p_d)

        r_c, r_d, r_f = poly_a.get_properties()

        self.assertTrue(len(r_c) == p_d + 1)
        self.assertTrue(r_d == p_d)
        self.assertTrue(isinstance(r_c[0], ZFieldElExt))
        self.assertTrue(r_f == ZUtils.FEXT)

        # inti poly by list
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        p_lc = [randint(1, prime * 2) for x in xrange(p_d + 1)]
        poly_a = ZPoly(p_lc)

        r_c, r_d, r_f = poly_a.get_properties()

        self.assertTrue(len(r_c) == p_d + 1)
        self.assertTrue(r_d == p_d)
        self.assertTrue(isinstance(r_c[0], ZFieldElExt))
        self.assertTrue(r_f == ZUtils.FEXT)

        # inti poly by list (ZFieldRdc)
        p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
        p_lc = [ZFieldElRedc(randint(1, prime * 2)) for x in xrange(p_d + 1)]
        poly_a = ZPoly(p_lc)

        r_c, r_d, r_f = poly_a.get_properties()

        self.assertTrue(len(r_c) == p_d + 1)
        self.assertTrue(r_d == p_d)
        self.assertTrue(isinstance(r_c[0], ZFieldElRedc))
        self.assertTrue(r_f == ZUtils.FRDC)

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
        self.assertTrue(r_f == ZUtils.FEXT)

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
        self.assertTrue(r_f == ZUtils.FRDC)

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
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(len(rrdc_c) == p_d + 1)
            self.assertTrue(rrdc_d == p_d)
            self.assertTrue(isinstance(rrdc_c[0], ZFieldElRedc))
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))

            # Pending Sparse Poly
            p_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
            n_coeff = randint(1, p_d)
            p_c = sample(xrange(p_d + 1), n_coeff)
            p_d = max(sorted(p_c))
            p_lc = {str(coeff_at): ZFieldElExt(randint(0, prime * 2)) for coeff_at in p_c}
            poly_a = ZPolySparse(p_lc)

            coeff = poly_a.get_coeff()
            coeff_ext = {c: coeff[c].extend() for c in coeff.keys()}
            coeff_rdc = {c: coeff[c].reduce() for c in coeff.keys()}

            poly_ext = ZPolySparse(coeff_ext)
            poly_rdc = ZPolySparse(coeff_rdc)

            rext_c, rext_d, rext_f = poly_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = poly_rdc.get_properties()

            self.assertTrue(len(rext_c) == p_d + 1)
            self.assertTrue(rext_d == p_d)
            self.assertTrue(isinstance(rext_c[0], ZFieldElExt))
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(len(rrdc_c) == p_d + 1)
            self.assertTrue(rrdc_d == p_d)
            self.assertTrue(isinstance(rrdc_c[0], ZFieldElRedc))
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))

    def test_2expand(self):
        c = ZUtils.CURVE_DATA['BN128']
        prime = c['prime']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])

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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(len(rrdc_c) == new_d + 1)
            self.assertTrue(rrdc_d == new_d)
            self.assertTrue(isinstance(rrdc_c[0], ZFieldElRedc))
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c[p_d + 1:] == [ZFieldElExt(0)] * (new_d - p_d))
            self.assertTrue(rext_c[:p_d + 1] == poly_ext.get_coeff()[:p_d + 1])
            self.assertTrue(rrdc_c[p_d + 1:] == [ZFieldElRedc(0)] * (new_d - p_d))
            self.assertTrue(rrdc_c[:p_d + 1] == poly_rdc.get_coeff()[:p_d + 1])

            #Expand operarion not defined for Sparse poly

    def test_3aritmetic(self):
        c = ZUtils.CURVE_DATA['BN128']
        prime = c['prime_r']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
        ZPoly(1,force_init=True)

        roots_ext, inv_roots_ext = ZField.find_roots(ZUtils.NROOTS, rformat_ext=True)
        roots_rdc = [r.reduce() for r in roots_ext]
        inv_roots_rdc = [r.reduce() for r in inv_roots_ext]

        # *, + , - , /, inv, scalar mul
        for i in xrange(ZPolyTest.TEST_ITER):
            p1_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
            p2_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
            shift_sc = randint(1, ZPolyTest.SHIFT_SCALAR)

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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
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

            # 
            p3_ext = poly1_ext << shift_sc
            p3_rdc = poly1_rdc << shift_sc

            rext_c, rext_d, rext_f = p3_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_rdc.get_properties()

            self.assertTrue(rext_d == poly1_ext.get_degree())
            self.assertTrue(rrdc_d == poly1_rdc.get_degree())
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(rext_c == [r2 << shift_sc for r2 in poly1_ext.get_coeff()])
            self.assertTrue(rrdc_c == [r2 << shift_sc for r2 in poly1_rdc.get_coeff()])

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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
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
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
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
            self.assertTrue(rrdca_f == ZUtils.FRDC)
            self.assertTrue(rrdcb_f == ZUtils.FRDC)
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
            self.assertTrue(rrdca_f == ZUtils.FRDC)
            self.assertTrue(rrdcb_f == ZUtils.FRDC)
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

            ZField.roots[0] = roots_ext
            ZField.inv_roots[0] = inv_roots_ext
            p1_ext.poly_mul(p2_ext)

            ZField.roots[0] = roots_rdc
            ZField.inv_roots[0] = inv_roots_rdc
            p1_rdc.poly_mul(p2_rdc)

            rext_c, rext_d, rext_f = p1_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p1_rdc.get_properties()


            poly1_ext.poly_mul_normal(poly2_ext)
            p3_c, p3_d, p3_f = poly1_ext.get_properties()

            self.assertTrue(rext_d == rrdc_d)
            self.assertTrue(rext_d == p3_d)
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue([c.as_long() for c in rext_c[:len(p3_c)]] == p3_c)

            # /
            pn_ext = ZPoly(p1_d)
            pd_ext = ZPoly(p1_d)
            pr_ext = ZPoly(p1_d/4)

            pn_rdc = pn_ext.reduce()
            pd_rdc = pd_ext.reduce()
            pr_rdc = pr_ext.reduce()

            ZField.roots[0] = roots_ext
            ZField.inv_roots[0] = inv_roots_ext
            pnum_ext = ZPoly(pn_ext)
            pden_ext = ZPoly(pd_ext)
            pnum_ext.poly_mul(pden_ext)
            pnum_ext += pr_ext
            q_ext    = pnum_ext.poly_div(pd_ext)

            ZField.roots[0] = roots_rdc
            ZField.inv_roots[0] = inv_roots_rdc
            pnum_rdc = ZPoly(pn_rdc)
            pden_rdc = ZPoly(pd_rdc)
            pnum_rdc.poly_mul(pden_rdc)
            pnum_rdc += pr_rdc
            q_rdc    = pnum_rdc.poly_div(pd_rdc)

            rext_c, rext_d, rext_f = q_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = q_rdc.get_properties()

            self.assertTrue(q_ext == pn_ext)
            self.assertTrue(q_rdc == pn_rdc)
            self.assertTrue(rext_d == pn_ext.get_degree())
            self.assertTrue(rrdc_d == pn_rdc.get_degree())
            self.assertTrue(len(rext_c) == rext_d+1)
            self.assertTrue(len(rext_c) == len(rrdc_c))
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(all([re.reduce() == rr for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(all([re == rr.extend() for re, rr in zip(rext_c, rrdc_c)]))
            self.assertTrue(rext_c == pn_ext.get_coeff())
            self.assertTrue(rrdc_c == pn_rdc.get_coeff())

    def test_4aritmetic_sparse(self):
        c = ZUtils.CURVE_DATA['BN128']
        prime = c['prime_r']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
        ZPoly.init()

        # *, + , - , /, inv, scalar mul
        for i in xrange(ZPolyTest.TEST_ITER):
            p1_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
            n_coeff = randint(1, p1_d)
            p_c = sample(xrange(p1_d + 1), n_coeff) + [p1_d]
            p_lc = {str(coeff_at): ZFieldElExt(randint(0, prime * 2)) for coeff_at in p_c}
            poly1_sps_ext = ZPolySparse(p_lc)
            poly1_dense_ext = poly1_sps_ext.to_dense()
            poly1_sps_rdc = poly1_sps_ext.reduce()
            poly1_dense_rdc = poly1_sps_rdc.to_dense()

            p2_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)
            n_coeff = randint(1, p2_d)
            p_c = sample(xrange(p2_d + 1), n_coeff) + [p2_d]
            p_lc = {str(coeff_at): ZFieldElExt(randint(0, prime * 2)) for coeff_at in p_c}
            poly2_sps_ext = ZPolySparse(p_lc)
            poly2_dense_ext = poly2_sps_ext.to_dense()
            poly2_sps_rdc = poly2_sps_ext.reduce()
            poly2_dense_rdc = poly2_sps_rdc.to_dense()
            
            p3_d = randint(1, ZPolyTest.MAX_POLY_DEGREE)

            poly3_dense_ext = ZPoly(p3_d)
            poly3_dense_rdc = poly3_dense_ext.reduce()

            scalar_l = randint(1,prime-1)
            scalar_bn = BigInt(scalar_l)
            scalar_ext = ZFieldElExt(scalar_bn)
            scalar_rdc = scalar_ext.reduce()

            shift_b = randint(0,ZPolyTest.SHIFT_SCALAR-1)

            # + Sparse + Sparse
            p3_sps_ext = poly1_sps_ext + poly2_sps_ext
            p3_sps_rdc = poly1_sps_rdc + poly2_sps_rdc

            r3_dense_ext = poly1_dense_ext + poly2_dense_ext

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_sps_rdc.get_properties()

            self.assertTrue(rext_d == rrdc_d)
            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(p3_sps_rdc.to_dense() == r3_dense_ext.reduce())
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(p3_sps_ext.reduce() == p3_sps_rdc)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))
            self.assertTrue(isinstance(p3_sps_rdc, ZPolySparse))

            # Sparse + Dense
            p3_dense_ext = poly1_sps_ext + poly3_dense_ext
            p3_dense_rdc = poly1_sps_rdc + poly3_dense_rdc

            r3_dense_ext = poly1_dense_ext + poly3_dense_ext

            rext_c, rext_d, rext_f = p3_dense_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_dense_rdc.get_properties()

            self.assertTrue(rext_d == rrdc_d)
            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_dense_ext == r3_dense_ext)
            self.assertTrue(p3_dense_rdc == r3_dense_ext.reduce())
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(p3_dense_ext.reduce() == p3_dense_rdc)
            self.assertTrue(isinstance(p3_dense_ext, ZPoly))
            self.assertTrue(isinstance(p3_dense_rdc, ZPoly))

            
            #  Dense + Sparse
            p3_dense_ext = poly3_dense_ext + poly1_sps_ext
            p3_dense_rdc = poly3_dense_rdc + poly1_sps_rdc

            r3_dense_ext = poly3_dense_ext + poly1_dense_ext

            rext_c, rext_d, rext_f = p3_dense_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_dense_rdc.get_properties()

            self.assertTrue(rext_d == rrdc_d)
            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_dense_ext == r3_dense_ext)
            self.assertTrue(p3_dense_rdc == r3_dense_ext.reduce())
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(p3_dense_ext.reduce() == p3_dense_rdc)
            self.assertTrue(isinstance(p3_dense_ext, ZPoly))
            self.assertTrue(isinstance(p3_dense_rdc, ZPoly))

            # -
            # Sparse - Sparse
            p3_sps_ext = poly1_sps_ext - poly2_sps_ext
            p3_sps_rdc = poly1_sps_rdc - poly2_sps_rdc

            r3_dense_ext = poly1_dense_ext - poly2_dense_ext

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_sps_rdc.get_properties()

            self.assertTrue(rext_d == rrdc_d)
            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(p3_sps_rdc.to_dense() == r3_dense_ext.reduce())
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(p3_sps_ext.reduce() == p3_sps_rdc)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))
            self.assertTrue(isinstance(p3_sps_rdc, ZPolySparse))

            # Sparse - Dense
            p3_dense_ext = poly1_sps_ext - poly3_dense_ext
            p3_dense_rdc = poly1_sps_rdc - poly3_dense_rdc

            r3_dense_ext = poly1_dense_ext - poly3_dense_ext

            rext_c, rext_d, rext_f = p3_dense_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_dense_rdc.get_properties()

            self.assertTrue(rext_d == rrdc_d)
            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_dense_ext == r3_dense_ext)
            self.assertTrue(p3_dense_rdc == r3_dense_ext.reduce())
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(p3_dense_ext.reduce() == p3_dense_rdc)
            self.assertTrue(isinstance(p3_dense_ext, ZPoly))
            self.assertTrue(isinstance(p3_dense_rdc, ZPoly))

            #  Dense - Sparse
            p3_dense_ext = poly3_dense_ext - poly1_sps_ext
            p3_dense_rdc = poly3_dense_rdc - poly1_sps_rdc

            r3_dense_ext = poly3_dense_ext - poly1_dense_ext

            rext_c, rext_d, rext_f = p3_dense_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_dense_rdc.get_properties()

            self.assertTrue(rext_d == rrdc_d)
            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_dense_ext == r3_dense_ext)
            self.assertTrue(p3_dense_rdc == r3_dense_ext.reduce())
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(p3_dense_ext.reduce() == p3_dense_rdc)
            self.assertTrue(isinstance(p3_dense_ext, ZPoly))
            self.assertTrue(isinstance(p3_dense_rdc, ZPoly))

            # neg
            p3_sps_ext = -poly2_sps_ext
            p3_sps_rdc = -poly2_sps_rdc

            r3_dense_ext = -poly2_dense_ext

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()
            rrdc_c, rrdc_d, rrdc_f = p3_sps_rdc.get_properties()

            self.assertTrue(rext_d == rrdc_d)
            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(p3_sps_rdc.to_dense() == r3_dense_ext.reduce())
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(p3_sps_ext.reduce() == p3_sps_rdc)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))
            self.assertTrue(isinstance(p3_sps_rdc, ZPolySparse))

            # * scalar
            #scalar_l = randint(1,prime-1)
            #scalar_bn = BigInt(scalar_l)
            #scalar_ext = ZFieldElExt(scalar_bn)
            #scalar_rdc = scalar_ext.reduce()
            # Sparse - Sparse
            p3_sps_ext =  scalar_l * poly2_sps_ext
            r3_dense_ext = scalar_l * poly2_dense_ext

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()

            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))

            p3_sps_ext =  poly2_sps_ext * scalar_l
            r3_dense_ext = poly2_dense_ext * scalar_l

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()

            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))

            p3_sps_ext =  scalar_bn * poly2_sps_ext
            r3_dense_ext = scalar_bn * poly2_dense_ext

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()

            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))

            p3_sps_ext =  poly2_sps_ext * scalar_bn
            r3_dense_ext = poly2_dense_ext * scalar_bn

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()

            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))

            p3_sps_ext =  scalar_ext * poly2_sps_ext
            r3_dense_ext = scalar_ext * poly2_dense_ext

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()

            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))

            p3_sps_ext =  poly2_sps_ext * scalar_ext
            r3_dense_ext = poly2_dense_ext * scalar_ext

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()

            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))

            p3_sps_rdc =  scalar_rdc * poly2_sps_rdc
            r3_dense_rdc = scalar_rdc * poly2_dense_rdc

            rrdc_c, rrdc_d, rrdc_f = p3_sps_rdc.get_properties()

            self.assertTrue(r3_dense_rdc.get_degree() == rrdc_d)
            self.assertTrue(p3_sps_rdc.to_dense() == r3_dense_rdc)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(isinstance(p3_sps_rdc, ZPolySparse))

            p3_sps_rdc =  poly2_sps_rdc * scalar_rdc
            r3_dense_rdc = poly2_dense_rdc * scalar_rdc

            rrdc_c, rrdc_d, rrdc_f = p3_sps_rdc.get_properties()

            self.assertTrue(r3_dense_rdc.get_degree() == rrdc_d)
            self.assertTrue(p3_sps_rdc.to_dense() == r3_dense_rdc)
            self.assertTrue(rrdc_f == ZUtils.FRDC)
            self.assertTrue(isinstance(p3_sps_rdc, ZPolySparse))

            # <<
            p3_sps_ext =  poly2_sps_ext << shift_b
            r3_dense_ext = poly2_dense_ext << shift_b

            rext_c, rext_d, rext_f = p3_sps_ext.get_properties()

            self.assertTrue(r3_dense_ext.get_degree() == rext_d)
            self.assertTrue(p3_sps_ext.to_dense() == r3_dense_ext)
            self.assertTrue(rext_f == ZUtils.FEXT)
            self.assertTrue(isinstance(p3_sps_ext, ZPolySparse))

    def test_5parallel_fft(self):
        c = ZUtils.CURVE_DATA['BN128']
        prime = c['prime_r']
        ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
        ZPoly(1,force_init=True)

        ZUtils.NROOTS=  ZPolyTest.FFT_PARALLEL_N ** 2 * \
              (1 << (ZPolyTest.MAX_FFT_NCOLS_W + ZPolyTest.MAX_FFT_NCOLS + ZPolyTest.MAX_FFT_NROWS_W + ZPolyTest.MAX_FFT_NROWS))
        roots_ext, inv_roots_ext = ZField.find_roots(ZUtils.NROOTS, rformat_ext=True)
        roots_rdc = [r.reduce() for r in roots_ext]
        inv_roots_rdc = [r.reduce() for r in inv_roots_ext]

        # *, + , - , /, inv, scalar mul
        for i in xrange(ZPolyTest.TEST_ITER):
            nr = 1 << randint(0, ZPolyTest.MAX_FFT_NROWS)
            nwr = 1 << randint(1, ZPolyTest.MAX_FFT_NROWS_W)
            nc =  1 << randint(0, ZPolyTest.MAX_FFT_NCOLS)
            nwc = 1 << randint(1, ZPolyTest.MAX_FFT_NCOLS_W)
            p_d = (ZPolyTest.FFT_PARALLEL_N * nr * nwr * ZPolyTest.FFT_PARALLEL_N * nc * nwc) - 1

            poly_ext = ZPoly(p_d)

            poly_rdc = poly_ext.reduce()

            # *
            p_rdc = ZPoly(poly_rdc)
            r_rdc = ZPoly(poly_rdc)

            ZField.roots[0] = roots_rdc
            ZField.inv_roots[0] = inv_roots_rdc
            p_rdc.ntt_parallel3D(ZPolyTest.FFT_PARALLEL_N * nr , nwr, ZPolyTest.FFT_PARALLEL_N * nc, nwc)
            r_rdc.ntt()

            self.assertTrue(p_rdc == r_rdc)

if __name__ == "__main__":
    unittest.main()
