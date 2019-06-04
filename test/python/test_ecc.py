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
// File name  : test_ecc.py
//
// Date       : 20/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//   ECC test
//
// TODO 
//    incorrect format  -> once asserts substituted by exceptions,
//         test incorrect formats can be done
// ------------------------------------------------------------------

"""
import sys
import unittest
from random import randint
from past.builtins import xrange

sys.path.append('../../src/python')

from bigint import *
from zfield import *
from ecc import *


class ECCTest(unittest.TestCase):
    TEST_ITER = 1000
    TEST_ITER2 = 20
    INT_THR = 13423
    LONG_THR = ZUtils.CURVE_DATA['BN128']['prime'] * 4
    FIND_N_PRIMES = 20
    MAX_PRIME = int(1e4)
    SMALL_PRIME_LIST = ZUtils.find_primes(randint(3, MAX_PRIME), MAX_PRIME, cnt=FIND_N_PRIMES)

    def test_0init_ext(self):
        c = ZUtils.CURVE_DATA['Secp112r1']
        ZField(c['prime'], c['factor_data'])
        p = c['prime']

        zel1 = [ZFieldElExt(randint(0, p - 1)) for x in range(2)]
        zel2 = [ZFieldElExt(randint(0, p - 1)) for x in range(2)]

        zel1_exta = [zel1[0] * c['curve_params']['Gx'], zel1[1] * c['curve_params']['Gy'], 1]
        zel2_exta = [zel2[0] * c['curve_params']['Gx'], zel2[1] * c['curve_params']['Gy'], 1]

        zel1_rdca = [zel1_exta[0].reduce(), zel1_exta[1].reduce()]
        zel2_rdca = [zel2_exta[0].reduce(), zel2_exta[1].reduce()]

        # Affine -> Projective
        zel1_extp = [zel1_exta[0], zel1_exta[1], zel1_exta[1]]
        zel2_extp = [zel2_exta[0], zel2_exta[1], zel1_exta[1]]

        zel1_rdcp = [z.reduce() for z in zel1_extp]
        zel2_rdcp = [z.reduce() for z in zel2_extp]

        # Affine -> JAcobian
        zel1_extj = [zel1_exta[0], zel1_exta[1], zel1_exta[1]]
        zel2_extj = [zel2_exta[0], zel2_exta[1], zel2_exta[1]]

        zel1_rdcj = [z.reduce() for z in zel1_extj]
        # Check curve is not initialized
        a, b = ECC.get_curve()

        self.assertTrue(ECC.is_curve_init() == False)
        self.assertTrue(a is None)
        self.assertTrue(b is None)

        ## Init ECC 
        # F ext
        p1_exta = ECCAffine(zel1_exta, curve=c['curve_params'], force_init=True)
        p1_extp = ECCProjective(zel1_extp, curve=c['curve_params'])
        p1_extj = ECCJacobian(zel1_extj, curve=c['curve_params'])

        a, b = ECC.get_curve()
        r1_exta = p1_exta.get_P()
        r1_extp = p1_extp.get_P()
        r1_extj = p1_extj.get_P()

        self.assertTrue(ECC.is_curve_init() == True)
        self.assertTrue(a.as_long() == c['curve_params']['a'])
        self.assertTrue(b.as_long() == c['curve_params']['b'])

        self.assertTrue([r.as_long() for r in r1_exta] == p1_exta.as_list())
        self.assertTrue([r.as_long() for r in r1_extp] == p1_extp.as_list())
        self.assertTrue([r.as_long() for r in r1_extj] == p1_extj.as_list())

        self.assertTrue(p1_exta.is_affine() == True)
        self.assertTrue(p1_exta.is_projective() == False)
        self.assertTrue(p1_exta.is_jacobian() == False)
        self.assertTrue(p1_exta.is_inf() == False)

        self.assertTrue(p1_extp.is_affine() == False)
        self.assertTrue(p1_extp.is_projective() == True)
        self.assertTrue(p1_extp.is_jacobian() == False)
        self.assertTrue(p1_extp.is_inf() == False)

        self.assertTrue(p1_extj.is_affine() == False)
        self.assertTrue(p1_extj.is_projective() == False)
        self.assertTrue(p1_extj.is_jacobian() == True)
        self.assertTrue(p1_extj.is_inf() == False)

        # Init ECC Afine object
        p1_exta = ECCAffine(zel1_exta)
        p2_exta = ECCAffine(p1_exta)

        p1_extp = ECCProjective(zel1_extp)
        p2_extp = ECCProjective(p1_extp)

        p1_extj = ECCJacobian(zel1_extj)
        p2_extj = ECCJacobian(p1_extj)

        self.assertTrue(p1_exta == p2_exta)
        self.assertTrue(p1_extp == p2_extp)
        self.assertTrue(p1_extj == p2_extj)

    def test_1conversion(self):
        c = ZUtils.CURVE_DATA['Secp112r1']
        ZField(c['prime'], c['factor_data'])
        p = c['prime']

        for test in xrange(ECCTest.TEST_ITER):
            zel1 = [ZFieldElExt(randint(0, p - 1)) for x in range(2)]
            zel2 = [ZFieldElExt(randint(0, p - 1)) for x in range(2)]

            # Affine
            zel1_exta = [zel1[0] * c['curve_params']['Gx'], zel1[1] * c['curve_params']['Gy'], 1]
            zel2_exta = [zel2[0] * c['curve_params']['Gx'], zel2[1] * c['curve_params']['Gy'], 1]

            zel1_rdca = [zel1_exta[0].reduce(), zel1_exta[1].reduce(), zel1_exta[1].reduce()]
            zel2_rdca = [zel2_exta[0].reduce(), zel2_exta[1].reduce(), zel1_exta[1].reduce()]

            # Affine -> Projective
            p1_exta = ECCAffine(zel1_exta, c['curve_params'], force_init=True)
            p1_extp = p1_exta.to_projective()
            p1_rdca = ECCAffine(zel1_rdca)
            p1_rdcp = p1_rdca.to_projective()
            p2_exta = ECCAffine(zel2_exta)
            p2_extp = p2_exta.to_projective()
            p2_rdca = ECCAffine(zel2_rdca)
            p2_rdcp = p2_rdca.to_projective()

            p3_exta = p1_exta + p2_exta
            p3_rdca = p1_rdca + p2_rdca
            p3_extp = p1_extp + p2_extp
            p3_rdcp = p1_rdcp + p2_rdcp

            self.assertTrue(p3_exta == p3_rdca.extend())
            self.assertTrue(p3_rdca == p3_exta.reduce())
            self.assertTrue(p3_extp == p3_rdcp.extend())
            self.assertTrue(p3_rdcp == p3_extp.reduce())

            self.assertTrue(p1_exta == p1_extp.to_affine())
            self.assertTrue(p1_exta.to_projective() == p1_extp)
            self.assertTrue(p1_rdca == p1_rdcp.to_affine())
            self.assertTrue(p1_rdca.to_projective() == p1_rdcp)

            # Affine -> JAcobian
            p1_exta = ECCAffine(zel1_exta, c['curve_params'])
            p1_extj = p1_exta.to_jacobian()
            p1_rdca = ECCAffine(zel1_rdca)
            p1_rdcj = p1_rdca.to_jacobian()
            p2_exta = ECCAffine(zel2_exta)
            p2_extj = p2_exta.to_jacobian()
            p2_rdca = ECCAffine(zel2_rdca)
            p2_rdcj = p2_rdca.to_jacobian()

            p3_exta = p1_exta + p2_exta
            p3_rdca = p1_rdca + p2_rdca
            p3_extj = p1_extj + p2_extj
            p3_rdcj = p1_rdcj + p2_rdcj

            self.assertTrue(p3_exta == p3_rdca.extend())
            self.assertTrue(p3_rdca == p3_exta.reduce())
            self.assertTrue(p3_extj == p3_rdcj.extend())
            self.assertTrue(p3_rdcj == p3_extj.reduce())

            self.assertTrue(p1_exta == p1_extj.to_affine())
            self.assertTrue(p1_exta.to_jacobian() == p1_extj)
            self.assertTrue(p1_rdca == p1_rdcj.to_affine())
            self.assertTrue(p1_rdca.to_jacobian() == p1_rdcj)

    def test_2operators(self):
        c = ZUtils.CURVE_DATA['BN128']
        ZField(c['prime'], c['factor_data'])
        p = c['prime']

        for test in xrange(ECCTest.TEST_ITER2):
            alpha_ext = ZFieldElExt(randint(0, p - 1))
            k = ZFieldElExt(1)

            # Affine
            zel1_exta = [k * c['curve_params']['Gx'], k * c['curve_params']['Gy'], 1]

            p1_exta = ECCAffine(zel1_exta, c['curve_params'], force_init=True)

            while True:
                z1 = ZFieldElExt(randint(0, p - 1))
                z2 = ZFieldElExt(randint(0, p - 1))
                p1_exta = z1 * p1_exta
                p2_exta = z2 * p1_exta
                if p1_exta != p2_exta:
                    break

            # Affine -> JAcobian (TODO)
            p1_extp = p1_exta.to_projective()
            p2_extp = p2_exta.to_projective()

            p1_rdca = p1_exta.reduce()
            p1_rdcp = p1_rdca.to_projective()
            p2_rdca = p2_exta.reduce()
            p2_rdcp = p2_rdca.to_projective()

            pzero_p = p1_extp.point_at_inf()
            pzero_a = p1_exta.point_at_inf()

            self.assertTrue(p1_exta.is_on_curve() == True)
            self.assertTrue(p2_exta.is_on_curve() == True)
            self.assertTrue(p1_rdca.is_on_curve() == True)
            self.assertTrue(p2_rdca.is_on_curve() == True)
            self.assertTrue(p1_extp.is_on_curve() == True)
            self.assertTrue(p2_extp.is_on_curve() == True)
            self.assertTrue(p1_rdcp.is_on_curve() == True)
            self.assertTrue(p2_rdcp.is_on_curve() == True)

            # Operators: +. -. neg, mul. double
            r_exta = p1_exta + p2_exta
            r_rdca = p1_rdca + p2_rdca
            r_extp = p1_extp + p2_extp
            r_rdcp = p1_rdcp + p2_rdcp

            # test __eq__
            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_projective() == r_extp)
            self.assertTrue(r_exta == r_extp.to_affine())
            self.assertTrue(r_exta.to_projective().reduce() == r_rdcp)
            self.assertTrue(r_exta.reduce().to_projective() == r_rdcp)
            self.assertTrue(r_exta == r_rdcp.to_affine().extend())
            self.assertTrue(r_exta == r_rdcp.extend().to_affine())

            # test ne
            self.assertTrue((r_exta.reduce() != r_rdca) == False)
            self.assertTrue((r_exta != r_rdca.extend()) == False)
            self.assertTrue((r_exta.to_projective() != r_extp) == False)
            self.assertTrue((r_exta != r_extp.to_affine()) == False)
            self.assertTrue((r_exta.to_projective().reduce() != r_rdcp) == False)
            self.assertTrue((r_exta.reduce().to_projective() != r_rdcp) == False)
            self.assertTrue((r_exta != r_rdcp.to_affine().extend()) == False)
            self.assertTrue((r_exta != r_rdcp.extend().to_affine()) == False)

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extp.is_on_curve() == True)
            self.assertTrue(r_rdcp.is_on_curve() == True)

            # Operators: + Zero
            r_exta = p1_exta + pzero_a
            r_rdca = p1_rdca + pzero_a.reduce()
            r_extp = p1_extp + pzero_p
            r_rdcp = p1_rdcp + pzero_p.reduce()

            # test __eq__
            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_projective() == r_extp)
            self.assertTrue(r_exta == r_extp.to_affine())
            self.assertTrue(r_exta.to_projective().reduce() == r_rdcp)
            self.assertTrue(r_exta.reduce().to_projective() == r_rdcp)
            self.assertTrue(r_exta == r_rdcp.to_affine().extend())
            self.assertTrue(r_exta == r_rdcp.extend().to_affine())

            self.assertTrue(r_exta == p1_exta)
            self.assertTrue(r_rdca == p1_rdca)
            self.assertTrue(r_extp == p1_extp)
            self.assertTrue(r_rdcp == p1_rdcp)

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extp.is_on_curve() == True)
            self.assertTrue(r_rdcp.is_on_curve() == True)
            # Operators: + Zero
            r_exta = pzero_a + p1_exta
            r_rdca = pzero_a.reduce() + p1_rdca
            r_extp = pzero_p + p1_extp
            r_rdcp = pzero_p.reduce() + p1_rdcp

            # test __eq__
            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_projective() == r_extp)
            self.assertTrue(r_exta == r_extp.to_affine())
            self.assertTrue(r_exta.to_projective().reduce() == r_rdcp)
            self.assertTrue(r_exta.reduce().to_projective() == r_rdcp)
            self.assertTrue(r_exta == r_rdcp.to_affine().extend())
            self.assertTrue(r_exta == r_rdcp.extend().to_affine())

            self.assertTrue(r_exta == p1_exta)
            self.assertTrue(r_rdca == p1_rdca)
            self.assertTrue(r_extp == p1_extp)
            self.assertTrue(r_rdcp == p1_rdcp)

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extp.is_on_curve() == True)
            self.assertTrue(r_rdcp.is_on_curve() == True)
            # sub
            r_exta = p1_exta - p2_exta
            r_rdca = p1_rdca - p2_rdca
            r_extp = p1_extp - p2_extp
            r_rdcp = p1_rdcp - p2_rdcp

            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_projective() == r_extp)
            self.assertTrue(r_exta == r_extp.to_affine())
            self.assertTrue(r_exta.to_projective().reduce() == r_rdcp)
            self.assertTrue(r_exta.reduce().to_projective() == r_rdcp)
            self.assertTrue(r_exta == r_rdcp.to_affine().extend())
            self.assertTrue(r_exta == r_rdcp.extend().to_affine())

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extp.is_on_curve() == True)
            self.assertTrue(r_rdcp.is_on_curve() == True)
            r_exta = -p1_exta
            r_rdca = -p1_rdca
            r_extp = -p1_extp
            r_rdcp = -p1_rdcp

            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_projective() == r_extp)
            self.assertTrue(r_exta == r_extp.to_affine())
            self.assertTrue(r_exta.to_projective().reduce() == r_rdcp)
            self.assertTrue(r_exta.reduce().to_projective() == r_rdcp)
            self.assertTrue(r_exta == r_rdcp.to_affine().extend())
            self.assertTrue(r_exta == r_rdcp.extend().to_affine())

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extp.is_on_curve() == True)
            self.assertTrue(r_rdcp.is_on_curve() == True)
            # double operation
            r_exta = p1_exta.double()
            r_rdca = p1_rdca.double()
            r_extp = p1_extp.double()
            r_rdcp = p1_rdcp.double()

            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_extp.reduce() == r_rdcp)
            self.assertTrue(r_extp == r_rdcp.extend())

            self.assertTrue(r_exta.to_projective() == r_extp)
            self.assertTrue(r_exta == r_extp.to_affine())

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extp.is_on_curve() == True)
            self.assertTrue(r_rdcp.is_on_curve() == True)
            # *
            r1_exta = p1_exta * alpha_ext
            r2_rdca = p1_rdca * alpha_ext
            r3_exta = alpha_ext * p1_exta
            r4_rdca = alpha_ext * p1_rdca
            r1_extp = p1_extp * alpha_ext
            r2_rdcp = p1_rdcp * alpha_ext
            r3_extp = alpha_ext * p1_extp
            r4_rdcp = alpha_ext * p1_rdcp

            self.assertTrue(r1_exta == r2_rdca.extend())
            self.assertTrue(r3_exta == r4_rdca.extend())
            self.assertTrue(r1_exta == r3_exta)

            self.assertTrue(r1_extp == r2_rdcp.extend())
            self.assertTrue(r3_extp == r4_rdcp.extend())
            self.assertTrue(r1_extp == r3_extp)

            self.assertTrue(r1_exta == r3_extp.to_affine())

            self.assertTrue(isinstance(r1_exta.get_P()[0], ZFieldElExt))
            self.assertTrue(isinstance(r2_rdca.get_P()[0], ZFieldElRedc))
            self.assertTrue(isinstance(r1_extp.get_P()[0], ZFieldElExt))
            self.assertTrue(isinstance(r2_rdcp.get_P()[0], ZFieldElRedc))

            self.assertTrue(r1_exta.is_on_curve() == True)
            self.assertTrue(r2_rdca.is_on_curve() == True)
            self.assertTrue(r3_exta.is_on_curve() == True)
            self.assertTrue(r4_rdca.is_on_curve() == True)
            self.assertTrue(r1_extp.is_on_curve() == True)
            self.assertTrue(r2_rdcp.is_on_curve() == True)
            self.assertTrue(r3_extp.is_on_curve() == True)
            self.assertTrue(r4_rdcp.is_on_curve() == True)

    def test_3operators_jacobian(self):
        c = ZUtils.CURVE_DATA['BN128']
        ZField(c['prime'], c['factor_data'])
        p = c['prime']

        for test in xrange(ECCTest.TEST_ITER2):
            alpha_ext = ZFieldElExt(randint(0, p - 1))
            k = ZFieldElExt(1)

            # Affine
            zel1_exta = [k * c['curve_params']['Gx'], k * c['curve_params']['Gy'], 1]

            p1_exta = ECCAffine(zel1_exta, c['curve_params'], force_init=True)
            while True:
                z1 = ZFieldElExt(randint(0, p - 1))
                z2 = ZFieldElExt(randint(0, p - 1))
                p1_exta = z1 * p1_exta
                p2_exta = z2 * p1_exta
                if p1_exta != p2_exta:
                    break

            # Affine -> JAcobian
            p1_extj = p1_exta.to_jacobian()
            p2_extj = p2_exta.to_jacobian()

            p1_rdca = p1_exta.reduce()
            p1_rdcj = p1_rdca.to_jacobian()
            p2_rdca = p2_exta.reduce()
            p2_rdcj = p2_rdca.to_jacobian()

            pzero_j = p1_extj.point_at_inf()
            pzero_a = p1_exta.point_at_inf()

            self.assertTrue(p1_exta.is_on_curve() == True)
            self.assertTrue(p2_exta.is_on_curve() == True)
            self.assertTrue(p1_rdca.is_on_curve() == True)
            self.assertTrue(p2_rdca.is_on_curve() == True)
            self.assertTrue(p1_extj.is_on_curve() == True)
            self.assertTrue(p2_extj.is_on_curve() == True)
            self.assertTrue(p1_rdcj.is_on_curve() == True)
            self.assertTrue(p2_rdcj.is_on_curve() == True)

            # Operators: +. -. neg, mul. double
            r_exta = p1_exta + p2_exta
            r_rdca = p1_rdca + p2_rdca
            r_extj = p1_extj + p2_extj
            r_rdcj = p1_rdcj + p2_rdcj

            # test __eq__
            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_jacobian() == r_extj)
            self.assertTrue(r_exta == r_extj.to_affine())
            self.assertTrue(r_exta.to_jacobian().reduce() == r_rdcj)
            self.assertTrue(r_exta.reduce().to_jacobian() == r_rdcj)
            self.assertTrue(r_exta == r_rdcj.to_affine().extend())
            self.assertTrue(r_exta == r_rdcj.extend().to_affine())

            # test ne
            self.assertTrue((r_exta.reduce() != r_rdca) == False)
            self.assertTrue((r_exta != r_rdca.extend()) == False)
            self.assertTrue((r_exta.to_jacobian() != r_extj) == False)
            self.assertTrue((r_exta != r_extj.to_affine()) == False)
            self.assertTrue((r_exta.to_jacobian().reduce() != r_rdcj) == False)
            self.assertTrue((r_exta.reduce().to_jacobian() != r_rdcj) == False)
            self.assertTrue((r_exta != r_rdcj.to_affine().extend()) == False)
            self.assertTrue((r_exta != r_rdcj.extend().to_affine()) == False)

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extj.is_on_curve() == True)
            self.assertTrue(r_rdcj.is_on_curve() == True)

            # Operators: + Zero
            r_exta = p1_exta + pzero_a
            r_rdca = p1_rdca + pzero_a.reduce()
            r_extj = p1_extj + pzero_j
            r_rdcj = p1_rdcj + pzero_j.reduce()

            # test __eq__
            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_jacobian() == r_extj)
            self.assertTrue(r_exta == r_extj.to_affine())
            self.assertTrue(r_exta.to_jacobian().reduce() == r_rdcj)
            self.assertTrue(r_exta.reduce().to_jacobian() == r_rdcj)
            self.assertTrue(r_exta == r_rdcj.to_affine().extend())
            self.assertTrue(r_exta == r_rdcj.extend().to_affine())

            self.assertTrue(r_exta == p1_exta)
            self.assertTrue(r_rdca == p1_rdca)
            self.assertTrue(r_extj == p1_extj)
            self.assertTrue(r_rdcj == p1_rdcj)

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extj.is_on_curve() == True)
            self.assertTrue(r_rdcj.is_on_curve() == True)
            # Operators: + Zero
            r_exta = pzero_a + p1_exta
            r_rdca = pzero_a.reduce() + p1_rdca
            r_extj = pzero_j + p1_extj
            r_rdcj = pzero_j.reduce() + p1_rdcj

            # test __eq__
            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_jacobian() == r_extj)
            self.assertTrue(r_exta == r_extj.to_affine())
            self.assertTrue(r_exta.to_jacobian().reduce() == r_rdcj)
            self.assertTrue(r_exta.reduce().to_jacobian() == r_rdcj)
            self.assertTrue(r_exta == r_rdcj.to_affine().extend())
            self.assertTrue(r_exta == r_rdcj.extend().to_affine())

            self.assertTrue(r_exta == p1_exta)
            self.assertTrue(r_rdca == p1_rdca)
            self.assertTrue(r_extj == p1_extj)
            self.assertTrue(r_rdcj == p1_rdcj)

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extj.is_on_curve() == True)
            self.assertTrue(r_rdcj.is_on_curve() == True)
            # sub
            r_exta = p1_exta - p2_exta
            r_rdca = p1_rdca - p2_rdca
            r_extj = p1_extj - p2_extj
            r_rdcj = p1_rdcj - p2_rdcj

            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_jacobian() == r_extj)
            self.assertTrue(r_exta == r_extj.to_affine())
            self.assertTrue(r_exta.to_jacobian().reduce() == r_rdcj)
            self.assertTrue(r_exta.reduce().to_jacobian() == r_rdcj)
            self.assertTrue(r_exta == r_rdcj.to_affine().extend())
            self.assertTrue(r_exta == r_rdcj.extend().to_affine())

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extj.is_on_curve() == True)
            self.assertTrue(r_rdcj.is_on_curve() == True)
            r_exta = -p1_exta
            r_rdca = -p1_rdca
            r_extj = -p1_extj
            r_rdcj = -p1_rdcj

            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_exta.to_jacobian() == r_extj)
            self.assertTrue(r_exta == r_extj.to_affine())
            self.assertTrue(r_exta.to_jacobian().reduce() == r_rdcj)
            self.assertTrue(r_exta.reduce().to_jacobian() == r_rdcj)
            self.assertTrue(r_exta == r_rdcj.to_affine().extend())
            self.assertTrue(r_exta == r_rdcj.extend().to_affine())

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extj.is_on_curve() == True)
            self.assertTrue(r_rdcj.is_on_curve() == True)
            # double operation
            r_exta = p1_exta.double()
            r_rdca = p1_rdca.double()
            r_extj = p1_extj.double()
            r_rdcj = p1_rdcj.double()

            self.assertTrue(r_exta.reduce() == r_rdca)
            self.assertTrue(r_exta == r_rdca.extend())
            self.assertTrue(r_extj.reduce() == r_rdcj)
            self.assertTrue(r_extj == r_rdcj.extend())

            self.assertTrue(r_exta.to_jacobian() == r_extj)
            self.assertTrue(r_exta == r_extj.to_affine())

            self.assertTrue(r_exta.is_on_curve() == True)
            self.assertTrue(r_rdca.is_on_curve() == True)
            self.assertTrue(r_extj.is_on_curve() == True)
            self.assertTrue(r_rdcj.is_on_curve() == True)
            # *
            r1_exta = p1_exta * alpha_ext
            r2_rdca = p1_rdca * alpha_ext
            r3_exta = alpha_ext * p1_exta
            r4_rdca = alpha_ext * p1_rdca
            r1_extj = p1_extj * alpha_ext
            r2_rdcj = p1_rdcj * alpha_ext
            r3_extj = alpha_ext * p1_extj
            r4_rdcj = alpha_ext * p1_rdcj

            self.assertTrue(r1_exta == r2_rdca.extend())
            self.assertTrue(r3_exta == r4_rdca.extend())
            self.assertTrue(r1_exta == r3_exta)

            self.assertTrue(r1_extj == r2_rdcj.extend())
            self.assertTrue(r3_extj == r4_rdcj.extend())
            self.assertTrue(r1_extj == r3_extj)

            self.assertTrue(r1_exta == r3_extj.to_affine())

            self.assertTrue(isinstance(r1_exta.get_P()[0], ZFieldElExt))
            self.assertTrue(isinstance(r2_rdca.get_P()[0], ZFieldElRedc))
            self.assertTrue(isinstance(r1_extj.get_P()[0], ZFieldElExt))
            self.assertTrue(isinstance(r2_rdcj.get_P()[0], ZFieldElRedc))

            self.assertTrue(r1_exta.is_on_curve() == True)
            self.assertTrue(r2_rdca.is_on_curve() == True)
            self.assertTrue(r3_exta.is_on_curve() == True)
            self.assertTrue(r4_rdca.is_on_curve() == True)
            self.assertTrue(r1_extj.is_on_curve() == True)
            self.assertTrue(r2_rdcj.is_on_curve() == True)
            self.assertTrue(r3_extj.is_on_curve() == True)
            self.assertTrue(r4_rdcj.is_on_curve() == True)

            #
            alpha_ext = randint(1, 100)
            r1_ext = p1_exta * alpha_ext
            r2_ext = np.sum([p1_exta]*alpha_ext)
            self.assertTrue(r1_ext == r2_ext)

    def test_4operators_jacobian_ecc2(self):
        c = ZUtils.CURVE_DATA['BN128']
        ZField(c['prime'], c['factor_data'])
        p = c['prime']

        for test in xrange(ECCTest.TEST_ITER2):
            alpha_ext = ZFieldElExt(randint(0, p - 1))
            k = ZFieldElExt(1)

            # Affine
            zelx_exta = [k * c['curve_params_g2']['Gx1'], k * c['curve_params_g2']['Gx2']]
            zely_exta = [k * c['curve_params_g2']['Gy1'], k * c['curve_params_g2']['Gy2']]
            zelz_exta = [1, 0]

            p1_exta = ECCJacobian([zelx_exta, zely_exta, zelz_exta], c['curve_params'], force_init=True)
            while True:
                z1 = ZFieldElExt(randint(0, p - 1))
                z2 = ZFieldElExt(randint(0, p - 1))
                p1_exta = z1 * p1_exta
                p2_exta = z2 * p1_exta
                if p1_exta != p2_exta:
                    break

            # Affine -> JAcobian
            p1_extj = p1_exta
            p2_extj = p2_exta

            p1_rdcj = p1_extj.reduce()
            p2_rdcj = p2_extj.reduce()

            pzero_j = p1_extj.point_at_inf()

            # Operators: +. -. neg, mul. double
            r_extj = p1_extj + p2_extj
            r_rdcj = p1_rdcj + p2_rdcj

            # test __eq__
            self.assertTrue(r_extj.reduce() == r_rdcj)
            self.assertTrue(r_extj == r_rdcj.extend())

            # test ne
            self.assertTrue((r_extj.reduce() != r_rdcj) == False)
            self.assertTrue((r_extj != r_rdcj.extend()) == False)

            # Operators: + Zero
            r_extj = p1_extj + pzero_j
            r_rdcj = p1_rdcj + pzero_j.reduce()

            # test __eq__
            self.assertTrue(r_extj.reduce() == r_rdcj)
            self.assertTrue(r_extj == r_rdcj.extend())

            self.assertTrue(r_extj == p1_extj)
            self.assertTrue(r_rdcj == p1_rdcj)

            # Operators: + Zero
            r_extj = pzero_j + p1_extj
            r_rdcj = pzero_j.reduce() + p1_rdcj

            # test __eq__
            self.assertTrue(r_extj.reduce() == r_rdcj)
            self.assertTrue(r_extj == r_rdcj.extend())

            self.assertTrue(r_extj == p1_extj)
            self.assertTrue(r_rdcj == p1_rdcj)

            # sub
            r_extj = p1_extj - p2_extj
            r_rdcj = p1_rdcj - p2_rdcj

            self.assertTrue(r_extj.reduce() == r_rdcj)
            self.assertTrue(r_extj == r_rdcj.extend())

            r_extj = -p1_extj
            r_rdcj = -p1_rdcj

            self.assertTrue(r_extj.reduce() == r_rdcj)
            self.assertTrue(r_extj == r_rdcj.extend())

            # double operation
            r_extj = p1_extj.double()
            r_rdcj = p1_rdcj.double()

            self.assertTrue(r_extj.reduce() == r_rdcj)
            self.assertTrue(r_extj == r_rdcj.extend())


            # *
            r1_extj = p1_extj * alpha_ext
            r2_rdcj = p1_rdcj * alpha_ext
            r3_extj = alpha_ext * p1_extj
            r4_rdcj = alpha_ext * p1_rdcj

            self.assertTrue(r1_extj == r2_rdcj.extend())
            self.assertTrue(r3_extj == r4_rdcj.extend())
            self.assertTrue(r1_extj == r3_extj)


            #
            alpha_ext = randint(1, 100)
            r1_ext = p1_exta * alpha_ext
            r2_ext = np.sum([p1_exta]*alpha_ext)
            self.assertTrue(r1_ext == r2_ext)

    def test_5is_on_curve(self):
        c = ZUtils.CURVE_DATA['BN128']
        ZField(c['prime'], c['factor_data'])
        p = c['prime']

        for test in xrange(ECCTest.TEST_ITER):
            alpha_ext = ZFieldElExt(randint(0, p - 1))

            # Affine
            zel1_exta = [alpha_ext * c['curve_params']['Gx'], alpha_ext * c['curve_params']['Gy'], 1]

            p1_exta = ECCAffine(zel1_exta, c['curve_params'], force_init=True)

            p1_extp = p1_exta.to_projective()
            p1_extj = p1_exta.to_jacobian()

            p1_rdca = p1_exta.reduce()
            p1_rdcp = p1_rdca.to_projective()
            p1_rdcj = p1_rdca.to_jacobian()

            pzero_j = p1_extj.point_at_inf()
            pzero_a = p1_exta.point_at_inf()
            pzero_p = p1_extp.point_at_inf()

            self.assertTrue(p1_exta.is_on_curve() == False)
            self.assertTrue(p1_rdca.is_on_curve() == False)
            self.assertTrue(p1_extp.is_on_curve() == False)
            self.assertTrue(p1_rdcp.is_on_curve() == False)
            self.assertTrue(p1_extj.is_on_curve() == False)
            self.assertTrue(p1_rdcj.is_on_curve() == False)
            self.assertTrue(pzero_a.is_on_curve() == False)
            self.assertTrue(pzero_p.is_on_curve() == False)
            self.assertTrue(pzero_j.is_on_curve() == False)


if __name__ == "__main__":
    unittest.main()
