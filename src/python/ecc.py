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
// File name  : ecc
//
// Date       : 14/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implements Elliptic Curve Arithmetic
//
// ------------------------------------------------------------------

"""

from abc import ABCMeta, abstractmethod

from zfield import *


class ECC(object):
    __metaclass__ = ABCMeta

    # point coordinate indexes
    X = 0
    Y = 1
    Z = 2

    constants_init = False
    one = [None, None]
    two = [None, None]
    three = [None, None]
    four = [None, None]
    eight = [None, None]

    a = [None, None]
    b = [None, None]

    def __init__(self, p=None, curve=None):
        """
          Constructor

          Parameters
          ----------
            p           : list of None, int, long, BigInt or ZFieldEl
                            of 2 (AFFINE) or 3(JACOBIAN/PROJECTIVE) elements
            curve           : Dictionary with curve coefficiennts a,b that define curve y^2=x^3+a*x + b. Keys :
              'a' : long/int
              'b' : long/int
        """
        if not ZField.is_init():
            assert True, "Finite field not initialized"
        elif curve is None and not ECC.is_curve_init():
            assert True, "Curve is not initialized"
        elif type(p) is list:
            p_l = p
            if ((isinstance(self, ECCAffine) and len(p) != 2) or
                (isinstance(self, ECCProjective) and len(p) != 3) or
                (isinstance(self, ECCJacobian) and len(p) != 3)):
                    assert True, "Coordinate format doesn't match with point format"
        elif isinstance(p,ECC):
            p_l = p.P
        elif p is None:
            p_l = None

        if curve is not None:
            if not isinstance(curve['a'],int) and not isinstance(curve['a'],long):
                assert True, "Unexpected curve parameters"
            elif not isinstance(curve['b'],int) and not isinstance(curve['b'],long):
                assert True, "Unexpected curve parameters"
            else:
                self.init_curve(curve['a'], curve['b'])

        # p can be a list of int, long, BigInt
        if p_l is None or p_l[ECC.X] is None:
            #self.P = p_l
            self.P = [None, None, None]
            self.FIDX = None
        elif isinstance(p_l[ECC.X], int) or isinstance(p_l[ECC.X], long) or isinstance(p_l[ECC.X], ZFieldElExt):
            self.P = [ZFieldElExt(x) for x in p_l]
            self.FIDX = ZUtils.FEXT
        elif isinstance(p_l[ECC.X], ZFieldElRedc):
            self.P = [ZFieldElRedc(x) for x in p_l]
            self.FIDX = ZUtils.FRDC
        else :
            assert True, "Unexpected format"

        if ECC.constants_init is False:
            ECC.one = [ZFieldElExt(1), ZFieldElExt(1).reduce()]
            ECC.two = [ZFieldElExt(2), ZFieldElExt(2).reduce()]
            ECC.three = [ZFieldElExt(3), ZFieldElExt(3).reduce()]
            ECC.four = [ZFieldElExt(4), ZFieldElExt(4).reduce()]
            ECC.eight = [ZFieldElExt(8), ZFieldElExt(8).reduce()]

            ECC.constants_init = True

    @classmethod
    def is_curve_init(cls):
        return ECC.a[ZUtils.FEXT] is not None and ECC.b[ZUtils.FEXT] is not None

    @classmethod
    def init_curve(cls, a, b):
        ECC.a = [ZFieldElExt(a), ZFieldElExt(a).reduce()]
        ECC.b = [ZFieldElExt(b), ZFieldElExt(b).reduce()]

    @classmethod
    def get_curve(cls):
        """
          Returns curve parameters a,b
        """
        return ECC.a[ZUtils.FEXT], ECC.b[ZUtils.FEXT]

    def get_P(self):
        """
          Returns point P
        """
        return self.P

    def is_affine(self):
        """
          True if coordinate format AFFINE
          False otherwise
        """
        if isinstance(self, ECCAffine):
            return True
        else:
            return False

    def is_projective(self):
        """
          True if coordinate format PROJECTIVE
          False otherwise
        """
        if isinstance(self, ECCProjective):
            return True
        else:
            return False

    def is_jacobian(self):
        """
          True if coordinate format JACOBIAN
          False otherwise
        """
        if isinstance(self, ECCJacobian):
            return True
        else:
            return False

    def is_inf(self):
        """
          True if point on infinite
          False otherwise
        """
        return self.P[ECC.X] is None or (len(self.P) == 3 and self.P[ECC.Z] == 0)

    def reduce(self):
        """
         Return new Elliptic curve point with coordinates expressed in Montgomert format
        """
        if self.is_inf() or isinstance(self.P[ECC.X], ZFieldElRedc):
            newP = self.P
        elif isinstance(self.P[ECC.X], ZFieldElExt):
            newP = [x.reduce() for x in self.P]
        else:
            assert True, "Unexpected data type"

        if isinstance(self, ECCProjective):
            return ECCProjective(newP)
        elif isinstance(self, ECCJacobian):
            return ECCJacobian(newP)
        elif isinstance(self, ECCAffine):
            return ECCAffine(newP)
        else:
            assert True, "Unexpected data type"

    def extend(self):
        """
         Return new Elliptic curve point with coordinates expressed in extended format
        """
        if self.is_inf() or isinstance(self.P[ECC.X], ZFieldElExt):
            newP = self.P
        if isinstance(self.P[ECC.X], ZFieldElRedc):
            newP = [x.extend() for x in self.P]
        else:
            assert True, "Unexpected data type"

        if isinstance(self, ECCProjective):
            return ECCProjective(newP)
        elif isinstance(self, ECCJacobian):
            return ECCJacobian(newP)
        elif isinstance(self, ECCAffine):
            return ECCAffine(newP)
        else:
            assert True, "Unexpected data type"

    def as_list(self):
        if not self.is_inf():
            return [p.as_long() for p in self.P]
        else:
            return [None] * len(self.P)

    def showP(self):
        print self.as_list()

    def same_format(self, P2):
        """
         returns True of self and P2 are same format of ECC coordinate representation
        """
        return type(self) == type(P2)

    @abstractmethod
    def __eq__(self, P2):
        """
         P1 == P2
        """
        pass

    def __ne__(self, P2):
        """
          P1 != P2
        """
        return not (self == P2)

    @abstractmethod
    def __add__(self, P2):
        """
          P1 + P2
        """
        pass

    @abstractmethod
    def __sub__(self, P2):
        """
          P1 - P2
        """
        pass

    @abstractmethod
    def __neg__(self):
        """
          -P1
        """
        pass

    def __mul__(self, alpha):
        """
          P1 * alpha
          TODO : pending += for ECC points and >>= for BigInt, SLINDING window
        """
        if isinstance(alpha, int) or isinstance(alpha, long):
            scalar = alpha
        elif isinstance(alpha, ZFieldElRedc):
            assert True, "Unexpected type"
        elif isinstance(alpha,BigInt):
            scalar = alpha.bignum
        else:
            assert True, "Unexpected type"

        if self.is_inf():
            return self.point_at_inf()

        if scalar < 0:
            scalar = -scalar

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

    @abstractmethod
    def double(self):
        """
         2 * P1
        """
        pass

    @abstractmethod
    def is_on_curve(self):
        """
          True of point is on curve
          False otherwise
        """
        pass

    @abstractmethod
    def point_at_inf(self):
        """"
          Return point at infinity
        """
        pass

class ECCAffine(ECC):
    """
      Curve : y^2 = x^3 + a*x + b
      Point on the curve : (x,y) st complies with curve above and
        x,y are part of finite field Fp
    """

    def __init__(self, p=None, curve=None,force_init=False):
        if force_init :
            ECC.constants_init = False
        ECC.__init__(self, p=p, curve=curve)

    def to_projective(self):
        """
          Convert Affine to Projective coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)
        """
        if self.is_inf():
            return ECCProjective([None, None, None])
        elif isinstance(self.P[ECC.X], ZFieldEl):
            return ECCProjective([self.P[ECC.X], self.P[ECC.Y], ECC.one[self.FIDX]])
        else:
            assert True, "Unexpected data type"

    def to_affine(self):
        """
          Converts point to AFFINE
        """
        return self

    def to_jacobian(self):
        """
          Convert Affine to Jacobian coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)
        """

        if self.is_inf():
            return ECCJacobian([None, None, None])
        elif isinstance(self.P[ECC.X], ZFieldEl):
            return ECCJacobian([self.P[ECC.X], self.P[ECC.Y], ECC.one[self.FIDX]])
        else:
            assert True, "Unexpected data type"

    def is_on_curve(self):
        """
          True of point is on curve
          False otherwise
        """
        return not self.is_inf() and \
               (self.P[ECC.Y] * self.P[ECC.Y]) == \
               (self.P[ECC.X] * self.P[ECC.X] * self.P[ECC.X]) + \
               (ECC.a[self.FIDX] * self.P[ECC.X]) + ECC.b[self.FIDX]

    # Arithmetic operators
    # +, - , neg, *
    def __add__(self, P2):
        """
          P1 + P2

        """
        if not isinstance(P2,ECCAffine):
            assert True, "Incorrect point format"
        elif not self.same_format(P2):
            assert True, "Finite Field represenation does not match"

        if self.is_inf():
            return P2
        elif P2.is_inf():
            return self
        elif self == P2:
            return self.double()
        elif self.P[ECC.X] == P2.P[ECC.X] and self.P[ECC.Y] != P2.P[ECC.Y]:
            return ECCAffine([None, None])
        else:
            s = (self.P[ECC.Y] - P2.P[ECC.Y]) * (self.P[ECC.X] - P2.P[ECC.X]).inv()
            rx = s * s - self.P[ECC.X] - P2.P[ECC.X]
            ry = s * (self.P[ECC.X] - rx) - self.P[ECC.Y]

        return ECCAffine([rx, ry])

    def __sub__(self, P2):
        """
          P1 - P2
          Check P2 is Affine.
        """
        return self + -P2

    def __neg__(self):
        """
        -P1
        """
        if self.is_inf():
            return self
        else:
            return ECCAffine([self.P[ECC.X], -self.P[ECC.Y]])

    # doubling operation
    def double(self):
        """
         2 * P1
         https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Affine_Coordinates
         1LD + 2M + 2S
        """
        if self.is_inf() or self.P[ECC.Y] == 0:
            return self.point_at_inf()

        X = self.P[ECC.X]
        Xsq = X * X
        Y = self.P[ECC.Y]

        a = ECC.a[self.FIDX]
        one = ECC.one[self.FIDX]
        two = ECC.two[self.FIDX]
        three = ECC.three[self.FIDX]

        if self.FIDX == ZUtils.FEXT:
            Y2 = Y << one
            X2 = X << one
            l = Xsq << one
            l += Xsq
        else:
            Y2 = Y * two
            X2 = X * two
            l = three * Xsq

        l += a
        l = l * Y2.inv()
        rx = l * l - X2
        ry = l *(X - rx) - Y

        return ECCAffine([rx, ry])

        # comparison operators

    def __eq__(self, P2):
        """
          P1 == P2
        """
        if self.is_inf() and P2.is_inf():
          return True
        elif not isinstance(P2, ECCAffine):
            assert True, "Unexpected data type"
        else:
            return (self.P[ECC.X], self.P[ECC.Y]) == (P2.P[ECC.X], P2.P[ECC.Y])

    def point_at_inf(self):
        """"
          Return point at infinity
        """
        newP =  ECCAffine([None, None])
        if self.FIDX is None:
            newP.FIDX = ZUtils.DEFAULT_IN_PFORMAT
        else:
            newP.FIDX = self.FIDX

        return newP

class ECCProjective(ECC):
    """
      Curve : y^2 = x^3 + a*x + b
      Point on the curve : (x,y) st complies with curve above and
        x,y are part of finite field Fp
    """

    def __init__(self, p=None, curve=None,force_init=False):
        if force_init :
           ECC.constants_init = False

        ECC.__init__(self, p=p, curve=curve)

    def to_projective(self):
        """
          Convert Projective to Projective coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)
        """
        return self

    def to_affine(self):
        """
          Converts point to AFFINE
        """
        if self.is_inf():
            return ECCAffine([None, None])
        elif isinstance(self.P[ECC.X], ZFieldEl):
            div = self.P[ECC.Z].inv()
            return ECCAffine([self.P[ECC.X] * div, self.P[ECC.Y] * div])
        else:
            assert True, "Unexpected data type"

    def to_jacobian(self):
        """
          Convert Projective to Jacobian coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)
         TODO
        """
        return self.to_affine().to_jacobian()

    def is_on_curve(self):
        """
          True of point is on curve
          False otherwise
        """
        return not self.is_inf() and \
               self.P[ECC.Y] * self.P[ECC.Y] * self.P[ECC.Z] == self.P[ECC.X] * self.P[ECC.X] * self.P[ECC.X] + \
               ECC.a[self.FIDX] * self.P[ECC.X] * self.P[ECC.Z] * self.P[ECC.Z] + \
               ECC.b[self.FIDX] * self.P[ECC.Z] * self.P[ECC.Z] * self.P[ECC.Z]

    # Arithmetic operators
    # +, - , neg, *
    def __add__(self, P2):
        """
          P1 + P2
        https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Standard_Projective_Coordinates
        12M + 2S
        """
        if not isinstance(P2,ECCProjective):
            assert True, "Incorrect point format"
        elif not self.same_format(P2):
            assert True, "Finite Field represenation does not match"

        if self.is_inf():
            return P2
        elif P2.is_inf():
            return self

        X1, Y1, Z1 = self.P
        X2, Y2, Z2 = P2.P

        U1 = Y2 * Z1
        U2 = Y1 * Z2
        V1 = X2 * Z1
        V2 = X1 * Z2

        if V1 == V2:
          if U1 != U2:
                return ECCProjective([None, None, None])
          else:
                return self.double()

        U = U1 - U2
        V = V1 - V2
        Usq = U**2
        Vsq = V**2
        Vcube = Vsq * V
        W = Z1 * Z2
        A = (Usq * W) - Vcube
      
        if self.FIDX == ZUtils.FEXT:
           A -= (Vsq * V2) << ECC.one[self.FIDX]
        else:
           A -= (Vsq * V2) * ECC.two[self.FIDX] 

        X3 = V * A
        Y3 = U * (Vsq * V2 - A) - Vcube*U2
        Z3 = Vcube * W

        return ECCProjective([X3, Y3, Z3])

    def __sub__(self, P2):
        """
          P1 - P2
          Check P2 is Projective
        """
        return self + -P2

    def __neg__(self):
        """
          -P1
        """
        if self.is_inf():
            return self
        else:
            return ECCProjective([self.P[ECC.X], -self.P[ECC.Y], self.P[ECC.Z]])

    def double(self):
        """
         2 * P1
        https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Standard_Projective_Coordinates
        7M + 5S / 7M + 3S
        """
        if self.is_inf() or self.P[ECC.Y] == 0:
            return self

        X,Y,Z = self.P

        a = ECC.a[self.FIDX]
        one = ECC.one[self.FIDX]
        two = ECC.two[self.FIDX]
        three = ECC.three[self.FIDX]
        four = ECC.four[self.FIDX]
        eight = ECC.eight[self.FIDX]

        S = Y * Z  # S = Y * Z
        Ssq = S * S  # Ssq = S*S
        Scube = Ssq * S  # Scube = S*S*S
        Ysq = Y * Y
        B = X * Y * S # B = X * Y * S

        if a == -three:  # W = 3*(X+Z)*(X-Z)
            W = (X + Z) *  (X - Z)
            if self.FIDX == ZUtils.FEXT:
                W1 = W << one
                W += W1
            else:
                W = three * W
        else:  # W = a*Z^2 + 3*X^2
            W = a * Z * Z
            Xsq = X * X
            if self.FIDX == ZUtils.FEXT:
                W1 = Xsq << one
                W +=  W1 + Xsq
            else:
                W1 = Xsq * three
                W += W1

        if self.FIDX == ZUtils.FEXT:
            H = W * W - (B << three)  # H = W^2 - 8*B
            rx = H * S << one       # X' = 2 * H * S
            # Y' = W*(4*B-H) - 8*Y^2*S^2
            ry = W * ( (B << two) - H) - (Ysq * Ssq << three)
            rz = Scube << three  # Z' =8 * S^3
        else:
            H = W * W - B * eight  # H = W^2 - 8*B
            rx = H * S * two  # X' = 2 * H * S
            # Y' = W*(4*B-H) - 8*Y^2*S^2
            ry = W * (B * four - H) - Ysq * Ssq * eight
            rz = Scube * eight  # Z' =8 * S^3

        return ECCProjective([rx, ry, rz])

    # comparison operators
    def __eq__(self, P2):
        """
          P1 == P2
        """
        if self.is_inf() and P2.is_inf():
            return True
        elif not isinstance(P2, ECCProjective):
            assert True, "Unexpected data type"
        else:
            return (self.P[ECC.X] * P2.P[ECC.Z], self.P[ECC.Y] * P2.P[ECC.Z]) == \
                   (P2.P[ECC.X] * self.P[ECC.Z], P2.P[ECC.Y] * self.P[ECC.Z])

    def point_at_inf(self):
        """"
          Return point at infinity
        """
        newP =  ECCProjective([None, None])
        if self.FIDX is None:
            newP.FIDX = ZUtils.DEFAULT_IN_PFORMAT
        else:
            newP.FIDX = self.FIDX

        return newP


class ECCJacobian(ECC):
    """
      Curve : y^2 = x^3 + a*x + b
      Point on the curve : (x,y) st complies with curve above and
        x,y are part of finite field Fp
    """

    def __init__(self, p=None, curve=None,force_init=False):
        if force_init :
             ECC.constants_init = False
        ECC.__init__(self, p=p, curve=curve)

    def to_projective(self):
        """
          Convert Jacobian to Projective coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)

        """
        return self.to_affine().to_projective()

    def to_affine(self):
        """
          Converts point to AFFINE
        """
        if self.is_inf():
            return ECCAffine([None, None])
        elif isinstance(self.P[ECC.X], ZFieldEl):
            zinv = self.P[ECC.Z].inv()
            zinv_sq = zinv * zinv
            zinv_cube = zinv_sq * zinv
            return ECCAffine([self.P[ECC.X] * zinv_sq, self.P[ECC.Y] * zinv_cube])
        else:
            assert True, "Unexpected data type"

    def to_jacobian(self):
        """
          Don't do anything
        """
        return self

    def is_on_curve(self):
        """
          True of point is on curve
          False otherwise
          TODO
        """
        return self.to_affine().is_on_curve()

    # Arithmetic operators
    # +, - , neg, *
    def __add__(self, P2):
        """
          P1 + P2
          12M + 4S
        """
        if not isinstance(P2,ECCJacobian):
            assert True, "Incorrect point format"
        elif not self.same_format(P2):
            assert True, "Finite Field represenation does not match"

        if self.is_inf():
            return P2
        elif P2.is_inf():
            return self

        X1, Y1, Z1 = self.P
        X2, Y2, Z2 = P2.P

        Z1sq = Z1 * Z1
        Z1cube = Z1sq * Z1
        Z2sq = Z2 * Z2
        Z2cube = Z2sq * Z2

        U1 = X1 * Z2sq   
        U2 = X2 * Z1sq   
        S1 = Y1 * Z2cube 
        S2 = Y2 * Z1cube 

        if U1 == U2:
            if S1 != S2:
                return self.point_at_inf()
            else:
                return self.double()

        H = U2 - U1   
        R = S2 - S1   
        Hsq = H * H
        Hcube = Hsq * H

        X3 = R * R - Hcube   
        if self.FIDX == ZUtils.FEXT:
          X3 = X3 - (U1 * (Hsq << ECC.one[self.FIDX]) )
        else: 
          X3 = X3 - (U1 * Hsq * ECC.two[self.FIDX])
 
        Y3 = R * (U1*Hsq - X3) - S1 * Hcube
        Z3 = H * Z1 *Z2

        return ECCJacobian([X3, Y3, Z3])


    def __sub__(self, P2):
        """
          P1 - P2
          Check P2 is Jacobian
        """
        return self + -P2

    def __neg__(self):
        """
          -P1
          TODO
        """
        if self.is_inf():
            return self
        else:
            return ECCJacobian([self.P[ECC.X], -self.P[ECC.Y], self.P[ECC.Z]])

    # doubling operation
    def double(self):
        """
         2 * P1

         4M + 6S / 4M + 4S
        """
        if self.is_inf() or self.P[ECC.Y] == 0:
            return self.point_at_inf()

        X,Y,Z = self.get_P()

        Ysq = Y * Y
        Ysqsq = Ysq * Ysq
        Zsq = Z * Z

        S = X * Ysq
        if self.FIDX == ZUtils.FEXT:
            S = (S << ECC.two[self.FIDX])
        else:
            S = S * ECC.four[self.FIDX]

        if ECC.a[self.FIDX] == -ECC.three[self.FIDX]:
            M1 = (X + Zsq) * (X - Zsq)
            if self.FIDX == ZUtils.FEXT:
                M  = (M1 << ECC.one[self.FIDX])
                M  = M + M1
            else:
                M  = ECC.three[self.FIDX] * M1
        else:
            M1 = X * X
            if self.FIDX == ZUtils.FEXT:
                M = M1 + (M1 << ECC.one[self.FIDX])
            else:
                M = ECC.three[self.FIDX] * M1
            M = M + ECC.a[self.FIDX] * Zsq * Zsq

        X3 = M * M
        Z3 = Y * Z
        if self.FIDX == ZUtils.FEXT:
            X3 = X3 - (S << ECC.one[self.FIDX])
            Y3 = M * (S - X3) - (Ysqsq << ECC.three[self.FIDX])
            Z3 = (Z3 << ECC.one[self.FIDX])
        else:
            X3 = X3 - (S * ECC.two[self.FIDX])
            Y3 = M * (S - X3) - (Ysqsq * ECC.eight[self.FIDX])
            Z3 = Z3 * ECC.two[self.FIDX]

        return ECCJacobian([X3, Y3, Z3])

    # comparison operators
    def __eq__(self, P2):
        """
          P1 == P2
        """
        if self.is_inf() and P2.is_inf():
            return True
        elif not isinstance(P2, ECCJacobian):
            assert True, "Unexpected data type"
        else:
            Z1sq = self.P[ECC.Z] * self.P[ECC.Z]
            Z1cube = Z1sq * self.P[ECC.Z]
            Z2sq = P2.P[ECC.Z] * P2.P[ECC.Z]
            Z2cube = Z2sq * P2.P[ECC.Z]

            return (self.P[ECC.X] * Z2sq, self.P[ECC.Y] * Z2cube) == \
                   (P2.P[ECC.X] * Z1sq, P2.P[ECC.Y] * Z1cube)

    def point_at_inf(self):
        """"
          Return point at infinity
        """
        newP =  ECCJacobian([None, None])

        if self.FIDX is None:
            newP.FIDX = ZUtils.DEFAULT_IN_PFORMAT
        else:
            newP.FIDX = self.FIDX

        return newP

def ECC_F1(p=None):
    if ZUtils.DEFAULT_IN_REP_FORMAT == ZUtils.AFFINE:
        return ECCAffine(p=p)
    elif ZUtils.DEFAULT_IN_REP_FORMAT == ZUtils.JACOBIAN:
        return ECCJacobian(p=p)
    elif ZUtils.DEFAULT_IN_REP_FORMAT == ZUtils.PROJECTIVE:
        return ECCProjective(p=p)
    else :
        assert True, "Unexpected type"


