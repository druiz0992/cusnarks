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

# NOTES:
#  Some parts of the code based from Poject Nayuki
# 
# Elliptic curve point addition in projective coordinates
# 
# Copyright (c) 2018 Project Nayuki. (MIT License)
# https://www.nayuki.io/page/elliptic-curve-point-addition-in-projective-coordinates
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# - The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
# - The Software is provided "as is", without warranty of any kind, express or
#   implied, including but not limited to the warranties of merchantability,
#   fitness for a particular purpose and noninfringement. In no event shall the
#   authors or copyright holders be liable for any claim, damages or other
#   liability, whether in an action of contract, tort or otherwise, arising from,
#   out of or in connection with the Software or the use or other dealings in the
#   Software.
# 

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : eccf2
//
// Date       : 27/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implements Elliptic Curve Arithmetic over F2 extension field
//
// ------------------------------------------------------------------

"""

from zfield import *
from ecc import *


class ECC_F2(object):

    def __init__(self, p1=None, p2=None,coor_fmt=None, curve=None):
        """
          Constructor

          Parameters
          ----------
            p1            : list of None, int, long, BigInt or ZFieldEl
                            of 2 (AFFINE) or 3(JACOBIAN/PROJECTIVE) elements
            p2
            curve           : Dictionary with curve coefficiennts a,b that define curve y^2=x^3+a*x + b. Keys :
              'a' : long/int
              'b' : long/int
        """
        if coor_fmt == None:
            coor_fmt = ZUtils.DEFAULT_IN_REP_FORMAT

        if coor_fmt == ZUtils.AFFINE:
           self.ecc1 = ECCAffine(p1,curve=curve)
           self.ecc2 = ECCAffine(p2)
        elif coor_fmt == ZUtils.PROJECTIVE:
           self.ecc1 = ECCProjective(p1,curve=curve)
           self.ecc2 = ECCProjective(p2)
        elif coor_fmt == ZUtils.JACOBIAN:
           self.ecc1 = ECCJacobian(p1,curve=curve)
           self.ecc2 = ECCJacobian(p2)

        else:
            assert False, "Unexpected data type"
          

    def get_P(self):
        """
          Returns point P
        """
        return self.ecc1.get_P(), self.ecc2.get_P()

    def is_inf(self):
        """
          True if point on infinite
          False otherwise
        """
        return self.ecc1.is_inf() and self.ecc2.is_inf() is None

    def reduce(self):
        """
         Return new Elliptic curve point with coordinates expressed in Montgomert format
        """
        newP1 = self.ecc1.reduce()
        newP2 = self.ecc2.reduce()

        if isinstance(newP1, ECCProjective):
            return ECC_F2(p1=newP1, p2=newP2, coor_fmt=ZUtils.PROJECTIVE)
        elif isinstance(newP1, ECCJacobian):
            return ECC_F2(p1=newP1, p2=newP2, coor_fmt=ZUtils.JACOBIAN)
        elif isinstance(newP1, ECCAffine):
            return ECC_F2(p1=newP1, p2=newP2, coor_fmt=ZUtils.AFFINE)
        else:
            assert False, "Unexpected data type"

    def extend(self):
        """
         Return new Elliptic curve point with coordinates expressed in extended format
        """
        newP1 = self.ecc1.extend()
        newP2 = self.ecc2.extend()

        if isinstance(newP1, ECCProjective):
            return ECC_F2(p1=newP1, p2=newP2, coor_fmt=ZUtils.PROJECTIVE)
        elif isinstance(newP1, ECCJacobian):
            return ECC_F2(p1=newP1, p2=newP2, coor_fmt=ZUtils.JACOBIAN)
        elif isinstance(newP1, ECCAffine):
            return ECC_F2(p1=newP1, p2=newP2, coor_fmt=ZUtils.AFFINE)
        else:
            assert False, "Unexpected data type"

    def as_list(self):
        if not self.is_inf():
            return self.ecc1.as_list(), self.ecc2.as_list()
        else:
            return self.point_at_inf()

    def to_projective(self):
        """
          Convert Affine to Projective coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)
        """
        if self.is_inf():
            return ECC_F2(coor_fmt=ZUtils.PROJECTIVE)
        elif isinstance(self.ecc1.get_P()[ECC.X], ZFieldEl):
            newP1, newP2 = self.get_P()
            return  ECC_F2(p1 = newP1, p2 = newP2,coor_fmt = ZUtils.PROJECTIVE)
        else:
            assert False, "Unexpected data type"

    def to_affine(self):
        """
          Converts point to AFFINE
        """
        pass

    def to_jacobian(self):
        """
          Convert Affine to Jacobian coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)
        """

        if self.is_inf():
            return ECC_F2(coor_fmt = ZUtils.JACOBIAN)
        elif isinstance(self.ecc1.get_P()[ECC.X], ZFieldEl):
            newP1, newP2 = self.get_P()
            return ECC_F2(p1=newP1, p2=newP2, coor_fmt = ZUtils.JACOBIAN)
        else:
            assert False, "Unexpected data type"

    def is_on_curve(self):
        """
          True of point is on curve
          False otherwise
        """
        return self.ecc1.is_on_curve() and self.ecc2.is_on_curve()

    # Arithmetic operators
    # +, - , neg, *
    def __add__(self, other):
        """
          P1 + P2

        """
        newP1 = self.ecc1 + other.ecc1
        newP2 = self.ecc2 + other.ecc2

        if isinstance(self.ecc1, ECCProjective) and isinstance (other.ecc1, ECCProjective):
            return ECC_F2(p1=newP1.get_P(), p2 = newP2.get_P(), coor_fmt = ZUtils.PROJECTIVE)
        elif isinstance(self.ecc1, ECCJacobian) and isinstance (other.ecc1, ECCJacobian):
            return ECC_F2(p1=newP1.get_P(), p2 = newP2.get_P(), coor_fmt = ZUtils.JACOBIAN)
        elif isinstance(self.ecc1, ECCAffine) and isinstance (other.ecc1, ECCAffine):
            return ECC_F2(p1=newP1.get_P(), p2 = newP2.get_P(), coor_fmt = ZUtils.AFFINE)
        else:
            assert False, "Unexpected data type"

    # doubling operation
    def double(self):
        """
         2 * P1
         https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Affine_Coordinates
         1LD + 2M + 2S
        """
        newP1 = self.ecc1.double()
        newP2 = self.ecc2.double()

        if isinstance(self.ecc1, ECCProjective):
            return ECC_F2(p1=newP1.get_P(), p2 = newP2.get_P(), coor_fmt = ZUtils.PROJECTIVE)
        elif isinstance(self.ecc1, ECCJacobian):
            return ECC_F2(p1=newP1.get_P(), p2 = newP2.get_P(), coor_fmt = ZUtils.JACOBIAN)
        elif isinstance(self.ecc1, ECCAffine):
            return ECC_F2(p1=newP1.get_P(), p2 = newP2.get_P(), coor_fmt = ZUtils.AFFINE)
        else:
            assert False, "Unexpected data type"

    def __mul__(self, alpha):
        """
          P1 * alpha
          TODO : pending += for ECC points and >>= for BigInt, SLINDING window
        """
        if isinstance(alpha, int) or isinstance(alpha, long):
            scalar = alpha
        elif isinstance(alpha, ZFieldElRedc):
            assert False, "Unexpected type"
        elif isinstance(alpha,BigInt):
            scalar = alpha.bignum
        else:
            assert False, "Unexpected type"

        if self.is_inf():
            return self.point_at_inf()

        if scalar < 0:
            scalar = -scalar
        elif scalar == 0:
            return self.point_at_inf(self)
        elif scalar == 1:
            return ECC_F2(p1=self.ecc1, p2=self.ecc2)
        elif self.is_inf():
            return self.point_at_inf(self)

        newP = self.point_at_inf()
        result = self
        while scalar != 0:
            if scalar & 1 != 0:
                newP += result
            result = result.double()
            scalar >>= 1
        return result

    def __rmul__(self, alpha):
        """
          alpha * P
          TODO : pending += for ECC points and >>= for BigInt, SLINDING window
        """
        return self * alpha

    def point_at_inf(self):
        """"
          Return point at infinity
        """
        if isinstance(self.ecc1, ECCProjective):
            return ECC_F2(coor_fmt = ZUtils.PROJECTIVE)
        elif isinstance(self.ecc1, ECCJacobian):
            return ECC_F2(coor_fmt = ZUtils.JACOBIAN)
        elif isinstance(self.ecc1, ECCAffine):
            return ECC_F2(coor_fmt = ZUtils.AFFINE)
        else:
            assert False, "Unexpected data type"

