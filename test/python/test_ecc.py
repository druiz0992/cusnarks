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
from random import randint
import unittest

import sys
sys.path.append('../../src/python')

from bigint import *
from zfield import *
from ecc import *

class ECCTest(unittest.TestCase):

    TEST_ITER = 1000
    INT_THR  = 13423
    LONG_THR = ZField.BN128_DATA['prime'] * 4
    FIND_N_PRIMES = 20
    MAX_PRIME = int(1e4)
    SMALL_PRIME_LIST = ZUtils.find_primes(randint(3,MAX_PRIME), MAX_PRIME, cnt=FIND_N_PRIMES)

    def test_0init_ext(self):
        c = ECC.CURVE_DATA['Secp112r1']
        ZField(c['prime'], c['factor_data'])

        zel1 = [ZFieldElExt(randint(0,p) for x in range(2)]
        zel2 = [ZFieldElExt(randint(0,p) for x in range(2)]

        # Affine
        zel1_exta = [zel1[0] * c['curve_params']['Gx'], zel1[1]['curve_params']['Gy']]
        zel2_exta = [zel2[0] * c['curve_params']['Gx'], zel2[1]['curve_params']['Gy']]

        zel1_rdca = [zel1_exta[0].reduce(), zel1_exta[1].reduce()] 
        zel2_rdca = [zel2_exta[0].reduce(), zel2_exta[1].reduce()] 

        #Projective
        zel1_extp = [zel1_exta[0].to_projective(). zel1_exta[1].to_projective()]
        zel2_extp = [zel2_exta[0].to_projective(). zel2_exta[1].to_projective()]

        zel1_rdcp = [zel1_extp[0].reduce(), zel1_extp[1].reduce()]
        zel2_rdcp = [zel1_extp[0].reduce(), zel1_extp[1].reduce()]
    
        #JAcobian
        zel1_extj = [zel1_exta[0].to_jacobian(). zel1_exta[1].to_jacobian()]
        zel2_extj = [zel2_exta[0].to_jacobian(). zel2_exta[1].to_jacobian()]

        zel1_rdcj = [zel1_extj[0].reduce(), zel1_extj[1].reduce()]
        zel2_rdcj = [zel1_extj[0].reduce(), zel1_extj[1].reduce()]
    
        # Check curve is not initialized
        a,b = ECC.get_curve()

        self.assertTrue(ECC.is_curve_init() == False)
        self.assertTrue(a is None)
        self.assertTrue(b is None)
       
        ## Init ECC 
        # F ext
        p1_exta = ECCAffine(zel1_exta,c['curve_params'])
        p1_extp = ECCProjective(zel1_extp,c['curve_params'])
        p1_extj = ECCJacobian(zel1_extj,c['curve_params'])

        a,b = ECC.get_curve()
        r1_exta = p1_exta.get_P()
        r1_extp = p1_extp.get_P()
        r1_extj = p1_extj.get_P()

        self.assertTrue(ECC.is_curve_init == True)
        self.assertTrue(a == c['a'])
        self.assertTrue(b == c['b'])

        self.assertTrue(r1_exta == p1_exta)
        self.assertTrue(r1_extp == p1_extp)
        self.assertTrue(r1_extj == p1_extj)

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

        p1_extp = ECCAffine(zel1_extp)
        p2_extp = ECCAffine(p1_extp)

        p1_extj = ECCAffine(zel1_extj)
        p2_extj = ECCAffine(p1_extj)

        self.assertTrue(p1_exta == p2_exta)
        self.assertTrue(p1_extp == p2_extp)
        self.assertTrue(p1_extj == p2_extj)

       #Init point at infinity
        p1_exta = ECCAffine([None, None])
        p1_extp = ECCAffine([None, None, None])
        p1_extj = ECCAffine([None, None, None])

        self.assertTrue(p1_exta.is_inf() == True)
        self.assertTrue(p1_exta.is_affine() == True)
        self.assertTrue(p1_exta.is_projective() == False)
        self.assertTrue(p1_exta.is_jacobian() == False)

        self.assertTrue(p1_extp.is_inf() == True)
        self.assertTrue(p1_extp.is_affine() == False)
        self.assertTrue(p1_extp.is_projective() == True)
        self.assertTrue(p1_extp.is_jacobian() == False)

        self.assertTrue(p1_extj.is_inf() == True)
        self.assertTrue(p1_extj.is_affine() == False)
        self.assertTrue(p1_extj.is_projective() == False)
        self.assertTrue(p1_extj.is_jacobian() == True)

    def test_1conversion(self):
        c = ECC.CURVE_DATA['Secp112r1']
        ZField(c['prime'], c['factor_data'])

        for test in xrange(ECCTest:TEST_ITER):
           zel1 = [ZFieldElExt(randint(0,p) for x in range(2)]
           zel2 = [ZFieldElExt(randint(0,p) for x in range(2)]

           # Affine
           zel1_exta = [zel1[0] * c['curve_params']['Gx'], zel1[1]['curve_params']['Gy']]
           zel2_exta = [zel2[0] * c['curve_params']['Gx'], zel2[1]['curve_params']['Gy']]
   
           zel1_rdca = [zel1_exta[0].reduce(), zel1_exta[1].reduce()] 
           zel2_rdca = [zel2_exta[0].reduce(), zel2_exta[1].reduce()] 

           #Affine -> Projective
           zel1_extp = [zel1_exta[0].to_projective(). zel1_exta[1].to_projective()]
           zel2_extp = [zel2_exta[0].to_projective(). zel2_exta[1].to_projective()]

           zel1_rdcp = [zel1_extp[0].reduce(), zel1_extp[1].reduce()]
           zel2_rdcp = [zel1_extp[0].reduce(), zel1_extp[1].reduce()]
   
           #Affine -> JAcobian
           zel1_extj = [zel1_exta[0].to_jacobian(). zel1_exta[1].to_jacobian()]
           zel2_extj = [zel2_exta[0].to_jacobian(). zel2_exta[1].to_jacobian()]

           zel1_rdcj = [zel1_extj[0].reduce(), zel1_extj[1].reduce()]
           zel2_rdcj = [zel1_extj[0].reduce(), zel1_extj[1].reduce()]

           zel3_exta = zel1_exta + zel2_exta
           zel3_rdca = zel1_rdca + zel2_rdca
           zel3_extj = zel1_extj + zel2_extj
           zel3_rdcj = zel1_rdcj + zel2_rdcj

           self.assertTrue(zel3_exta == zel3_rdca.extend())
           self.assertTrue(zel3_rdca == zel3_exta.reduce())

           self.assertTrue(zel3_exta == zel3_extp.to_affine())
           self.assertTrue(zel3_exta.to_projective() == zel3_extp)
           self.assertTrue(zel1_exta == zel1_extj.to_affine())
           self.assertTrue(zel1_exta.to_jacobian() == zel1_extj)

           self.assertTrue(zel3_exta == zel3_rdcp.to_affine().extend())
           self.assertTrue(zel3_exta == zel3_rdcp.extend().to_affine())
           self.assertTrue(zel3_exta.to_projective().reduce() == zel3_rdcp)
           self.assertTrue(zel3_exta.reduce().to_projective() == zel3_rdcp)

           self.assertTrue(zel1_exta == zel1_rdcj.to_affine().extend())
           self.assertTrue(zel1_exta == zel1_rdcj.extend().to_affine())
           self.assertTrue(zel1_exta.to_projective().reduce() == zel1_rdcj)
           self.assertTrue(zel1_exta.reduce().to_projective() == zel1_rdck)


           # Define affine point at infinity 
           zel1_exta = [None, None]
           zel1_extp = [None, None, None]

           self.assertTrue(zel1_exta == zel1_extp)
           self.assertTrue((zel1_exta != zel1_extp) == False)
           self.assertTrue(zel1.exta.to_projective() == zel1_extp)
           self.assertTrue(zel1.extp.to_affine() == zel1_exta)


  def test_2arithmetic(self):
        c = ECC.CURVE_DATA['Secp112r1']
        ZField(c['prime'], c['factor_data'])

        for test in xrange(ECCTest:TEST_ITER):
           zel1 = [ZFieldElExt(randint(0,p) for x in range(2)]
           zel2 = [ZFieldElExt(randint(0,p) for x in range(2)]
           alpha_ext = ZFieldElExt(randint(0,p)
           alpha_rdc = ZFieldElRedc(randint(0,p)

           # Affine
           zel1_exta = [zel1[0] * c['curve_params']['Gx'], zel1[1]['curve_params']['Gy']]
           zel2_exta = [zel2[0] * c['curve_params']['Gx'], zel2[1]['curve_params']['Gy']]
   
           zel1_rdca = [zel1_exta[0].reduce(), zel1_exta[1].reduce()] 
           zel2_rdca = [zel2_exta[0].reduce(), zel2_exta[1].reduce()] 

           #Affine -> Projective
           zel1_extp = [zel1_exta[0].to_projective(). zel1_exta[1].to_projective()]
           zel2_extp = [zel2_exta[0].to_projective(). zel2_exta[1].to_projective()]

           zel1_rdcp = [zel1_extp[0].reduce(), zel1_extp[1].reduce()]
           zel2_rdcp = [zel1_extp[0].reduce(), zel1_extp[1].reduce()]
   
           #Affine -> JAcobian 
           zel1_extj = [zel1_exta[0].to_jacobian(). zel1_exta[1].to_jacobian()]
           zel2_extj = [zel2_exta[0].to_jacobian(). zel2_exta[1].to_jacobian()]

           zel1_rdcj = [zel1_extj[0].reduce(), zel1_extj[1].reduce()]
           zel2_rdcj = [zel1_extj[0].reduce(), zel1_extj[1].reduce()]

           # Operators: +. -. neg, mul. double
           r_exta = zel1_exta + zel2_exta
           r_rdca = zel1_rdca + zel2_rdca
           r_extp = zel1_extp + zel2_extp
           r_rdcp = zel1_rdcp + zel2_rdcp
    
           self.assertTrue(r_exta.reduce() == r_rdca)
           self.assertTrue(r_exta == r_rdca.extend())
           self.assertTrue(r_exta.to_projective() == r_extp)
           self.assertTrue(r_exta == r_extp.to_affine())
           self.assertTrue(r_.exta.to_projective().reduce() == r_rdcp)
           self.assertTrue(r_exta.reduce().to_projective() == r_rdcp)
           self.assertTrue(r_exta == r_rdcp.to_affine().extend())
           self.assertTrue(r_exta == r_rdcp.extend().to_affine())
           

           r_exta = zel1_exta - zel2_exta
           r_rdca = zel1_rdca - zel2_rdca
           r_extp = zel1_extp - zel2_extp
           r_rdcp = zel1_rdcp - zel2_rdcp

           self.assertTrue(r_exta.reduce() == r_rdca)
           self.assertTrue(r_exta == r_rdca.extend())
           self.assertTrue(r_exta.to_projective() == r_extp)
           self.assertTrue(r_exta == r_extp.to_affine())
           self.assertTrue(r_exta.to_projective().reduce() == r_rdcp)
           self.assertTrue(r_exta.reduce().to_projective() == r_rdcp)
           self.assertTrue(r_exta == r_rdcp.to_affine().extend())
           self.assertTrue(r_exta == r_rdcp.extend().to_affine())

           r_exta = -zel1_exta
           r_rdca = -zel1_rdca
           r_extp = -zel1_extp
           r_rdcp = -zel1_rdcp

           self.assertTrue(r_exta.reduce() == r_rdca)
           self.assertTrue(r_exta == r_rdca.extend())
           self.assertTrue(r_exta.to_projective() == r_extp)
           self.assertTrue(r_exta == r_extp.to_affine())
           self.assertTrue(r_exta.to_projective().reduce() == r_rdcp)
           self.assertTrue(r_exta.reduce().to_projective() == r_rdcp)
           self.assertTrue(r_exta == r_rdcp.to_affine().extend())
           self.assertTrue(r_exta == r_rdcp.extend().to_affine())

           r1_exta = zel1_exta * alpha_ext
           r2_rdca = zel1_exta * alpha_rdc
           r3_exta = alpha_ext * zel1_exta
           r4_rdca = alpha_rdc * zel1_exta
           r1_extp = zel1_extp * alpha_ext
           r2_rdcp = zel1_extp * alpha_rdc
           r3_extp = alpha_ext * zel1_extp
           r4_rdcp = alpha_rdc * zel1_extp

           self.assertTrue(r1_exta == r2_rdca.extend() == r3_exta == r4_rdca.extend())
           self.assertTrue(r1_exta.reduce() == r2_rdca == r3_exta.reduce() == r4_rdca)
           self.assertTrue(r1_extp == r2_rdcp.extend() == r3_extp == r4_rdcp.extend())
           self.assertTrue(r1_extp.reduce() == r2_rdcp == r3_extp.reduce() == r4_rdcp)
           self.assertTrue(r1_exta = r1_extp.to_affine())
           self.assertTrue(r1_exta.to_projective() = r1_extp)
           self.assertTrue(isinstance(r1_exta.get_P()[0],ZFieldElExt)
           self.assertTrue(isinstance(r1_rdca.get_P()[0],ZFieldElRedc)
           self.assertTrue(isinstance(r1_extp.get_P()[0],ZFieldElExt)
           self.assertTrue(isinstance(r1_rdcp.get_P()[0],ZFieldElRedc)

           r_exta = zel1_exta.double()
           r_rdca = zel1_rdca.double()
           r_extp = zel1_extp.double()
           r_rdcp = zel1_rdcp.double()
     
           self.assertTrue(r_exta.reduce() == r_rdca)
           self.assertTrue(r_exta == r_rdca.extend())
           self.assertTrue(r_exta.to_projective() == r_extp)
           self.assertTrue(r_exta == r_extp.to_affine())
           self.assertTrue(r_exta.to_projective().reduce() == r_rdcp)
           self.assertTrue(r_exta.reduce().to_projective() == r_rdcp)
           self.assertTrue(r_exta == r_rdcp.to_affine().extend())
           self.assertTrue(r_exta == r_rdcp.extend().to_affine())


if __name__ == "__main__":
    unittest.main()

