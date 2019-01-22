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

    FEXT = 0
    FRDC = 1

    constants_init = False
    one = [None, None]
    two = [None, None]
    three = [None, None]
    eight = [None, None]

    a = [None, None]
    b = [None, None]

    def __init__(self, p, curve=None):
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
        tmp_p = p
        if not isinstance(p,ECC) or type(p) is not list or (length(p) < 2 and length(p) != 0) or length(p) > 3:
            assert True, "Invalid point format. Expected a list"
        elif ((isinstance(self, ECCAffine) and len(p) != 2) or
              (isinstance(self, ECCProjective) and len(p) != 3) or
              (isinstance(self, ECCJacobian) and len(p) != 3)):
            assert True, "Coordinate format doesn't match with point format"
        elif not ZField.is_init():
            assert True, "Finite field not initialized"
        elif curve is None and not ECC.is_curve_init():
            assert True, "Curve is not initialized"
        elif curve is not None:
            if not isinstance(int, curve['a']) and not isinstance(long, curve['a']):
                assert True, "Unexpected curve parameters"
            elif not isinstance(int, curve['b']) and not isinstance(long, curve['b']):
                assert True, "Unexpected curve parameters"
            else:
                self.init_curve(curve['a'], curve['b'])
        else:
            if isinstance(p,ECC):
                tmp_p = p.P
            # p can be a list of int, long, BigInt
            if isinstance(p2[ECC.X], int) or isinstance(p2[ECC.X], long) or isinstance(p2[ECC.X], ZFieldElExt):
                self.P = [ZFieldElExt(x) for x in tmp_p]
                self.FIDX = FEXT
            elif isinstance(p[ECC.X], ZFieldElRedc):
                self.P = [ZFieldElRedc(x) for x in tmp_p]
                self.FIDX = FRDC
            else:
                assert True, "Unexpected format"

        if ECC.constants_init is False:
            ECC.one = [ZFieldElExt(1), ZFieldElRedc(1)]
            ECC.two = [ZFieldElExt(2), ZFieldElRedc(2)]
            ECC.three = [ZFieldElExt(3), ZFieldElRedc(3)]
            ECC.eight = [ZFieldElExt(8), ZFieldElRedc(8)]

            ECC.constants_init = True

    @classmethod
    def is_curve_init(cls):
        return ECC.a[ECC.FEXT] is not None and ECC.b[ECC.FEXT] is not None

    @classmethod
    def init_curve(cls, a, b):
        ECC.a = [ZFieldElExt(a), ZFieldElRedc(a)]
        ECC.b = [ZFieldElExt(b), ZFieldElRedc(b)]

    @classmethod
    def get_curve(cls):
        """
          Returns curve parameters a,b
        """
        return ECC.a[ECC.FEXT], ECC.b[ECC.FEXT]

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
        return self.P[ECC.X] is None

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
        return [p.as_long() for p in self.P]

    def showP(self):
        print self.as_list()

    def same_format(self, P2):
        """
         returns True of self and P2 are same format of ECC coordinate representation
        """
        return type(self) == type(p2c)

    @abstractmethod
    def __eq__(self, P2):
        """
         P1 == P2
        """
        pass

    @abstractmethod
    def __ne__(self, P2):
        """
          P1 != P2
        """
        pass

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

    @abstractmethod
    def __mul__(self, alpha):
        """
          alpha * P1
        """
        pass

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


class ECCAffine(ECC):
    """
      Curve : y^2 = x^3 + a*x + b
      Point on the curve : (x,y) st complies with curve above and
        x,y are part of finite field Fp
    """

    def __init__(self, p, curve=None):
        ECC.__init__(self, p, curve)

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
        pass

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
               (self.P[ECC.Y] * self.P[ECC.Y]).extend() == \
               (self.P[ECC.X] * self.P[ECC.X].extend()).extend() + \
               (ECC.a[self.FIDX] * self.P[ECC.X]).extend() + ECC.b[ECC_FIDX]

    # Arithmetic operators
    # +, - , neg, *
    def __add__(self, P2):
        """
          P1 + P2

        """
        if not isinstance(ECCAffine, P2):
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
            return ([self.P[ECC.X], -self.P[ECC.Y]])

    def __mul__(self, alpha):
        """
          alpha * P1
          TODO : SLIDING window
        """
        if isinstance(alpha, int) or isinstance(alpha, long):
            scalar = alpha
        elif isinstance(alpha, BigInt):
            scalar = alpha.bignum

        if self.is_inf():
            return ECCAffine([None, None])

        if scalar < 0:
            scalar = -scalar

        newP = ECCAffine([None, None])
        temp = self
        while scalar != 0:
            if scalar & 1 != 0:
                newP += temp
            temp = temp.double()
            scalar >>= 1
        return result

    # doubling operation
    def double(self):
        """
         2 * P1
        """
        if self.is_inf() or self.P[ECC.Y] == 0:
            return self

        s = self.P[ECC.X] * self.P[ECC.X]
        if self.FIDX == ECC.FEXT:
            s += (self.P[ECC.X] * self.P[ECC.X]) << ECC.one[self.FIDX]
            s += ECC.a[self.FIDX]
            s += (self.P[ECC.Y] << ECC.one[self.FIDX]).inv()
            rx = s * s - (self.P[ECC.X] << ECC.one[self.FIDX])
        else:
            s += (self.P[ECC.X] * self.P[ECC.X]) * ECC.two[self.FIDX]
            s += ECC.a[self.FIDX]
            s += (self.P[ECC.Y] * ECC.two[self.FIDX]).inv()
            rx = s * s - (self.P[ECC.X] * ECC.two[self.FIDX])

        ry = s * (self.P[ECC.X] - rx) - self.P[ECC.Y]

        return ECCAffine([rx, ry])

        # comparison operators

    def __eq__(self, P2):
        """
          P1 == P2
        """
        if not isinstance(P2, ECCAffine):
            assert True, "Unexpected data type"
        elif self.is_zero() and P2.is_zero():
            return True
        else:
            return (self.P[ECC.X], self.P[ECC.Y]) == (P2.P[ECC.X], P2.P[ECC.Y])

    def __ne__(self, P2):
        """
          True if points are different
          False otherwis
        """
        return not (self == P2)


class ECCProjective(ECC):
    """
      Curve : y^2 = x^3 + a*x + b
      Point on the curve : (x,y) st complies with curve above and
        x,y are part of finite field Fp
    """

    def __init__(self, p, curve=None):
        ECC.__init__(self, p, curve)

    def to_projective(self):
        """
          Convert Projective to Projective coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)
        """
        pass

    def to_affine(self):
        """
          Converts point to AFFINE
        """
        if self.is_inf():
            return ECCProjective([None, None])
        elif isinstance(self.P[ECC.X], ZFieldEl):
            div = ZField.inv(self.P[ECC.Z])
            return ECCProjective([self.P[ECC.X] * div, self.P[ECC.Y] * div])
        else:
            assert True, "Unexpected data type"

    def to_projective(self):
        """
          Don't do eanything
        """
        pass

    def to_jacobian(self):
        """
          Convert Projective to Jacobian coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)
         TODO
        """
        pass

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

        """
        if not isinstance(ECCProjective, P2):
            assert True, "Incorrect point format"
        elif not self.same_format(P2):
            assert True, "Finite Field represenation does not match"

        if self.is_inf():
            return P2
        elif P2.is_inf():
            return self

        t0 = self.P[ECC.Y] * P2.P[ECC.Z]
        t1 = P2.P[ECC.Y] * self.P[ECC.Z]
        u0 = self.P[ECC.X] * P2.P[ECC.Z]
        u1 = P2.P[ECC.X] * self.P[ECC.Z]

        if u0 == u1:
            if t0 == t1:
                return self.double()
            else:
                return ECCProjective([None, None, None])
        else:
            t = t0 - t1
            u = u0 - u1
            u2 = u * u
            v = self.P[ECC.Z] * P2.P[ECC.Z]
            w = t * t * v - u2 * (u0 + u1)
            u3 = u = u2
            rx = u * w
            ry = t * (u0 * u2 - w) - t0 * u3
            rz = u3 * v
        return ECCProjective([rx, ry, rz])

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
            return ([self.P[ECC.X], -self.P[ECC.Y]], self.P[ECC.Z])

    def __mul__(self, alpha):
        """
          alpha * P1
          TODO : pending += for ECC points and >>= for BigInt, SLINDING window
        """
        if isinstance(alpha, int) or isinstance(alpha, long):
            scalar = alpha
        elif isinstance(alpha, BigInt):
            scalar = alpha.bignum

        if self.is_inf():
            return ECCProjective([None, None, None])

        if scalar < 0:
            scalar = -scalar

        newP = ECCProjective([None, None, None])
        temp = self
        while scalar != 0:
            if scalar & 1 != 0:
                newP += temp
            temp = temp.double()
            scalar >>= 1
        return result

    # doubling operation
    def double(self):
        """
         2 * P1
        """
        if self.is_inf() or seld.P[ECC.Y] == 0:
            return self

        S = self.P[ECC.Y] * self.P[ECC.Z]  # S = Y * Z
        S2 = S * S  # S2 = S*S
        B = self.P[ECC.X] * self.P[ECC.Y] * S  # B = X * Y * S

        if ECC.a[self.FIDX] == ECC.three[self.FIDX]:  # W = 3*(X+Z)*(X-Z)
            W = (self.P[ECC.X] + self.P[ECC.Z]) * \
                (self.P[ECC.X] - self.P[ECC.Z])

            if self.FIDX == ECC.FEXT:
                W1 = W << ECC.one[self.FIDX]
                W += W1
            else:
                W = ECC.three[self.FIDX] * W
        else:  # W = a*Z^2 + 3*X^2
            W = ECC.a[ECC_FIDX] * self.P[ECC.Z] * self.P[ECC.Z]
            X2 = self.P[ECC.X] * self.P[ECC.X]
            if self.FIDX == ECC.FEXT:
                W1 = X2 << ECC.one[self.FIDX]
                W = W + W1 + X2
            else:
                W1 = X2 * ECC.three[self.FIDX]
                W = W + W1

        if self.FIDX == ECC.FEXT:
            H = W * W - B << ECC.three[self.FIDX]  # H = W^2 - 8*B
            rx = H * S << ECC.one[self.FIDX]  # X' = 2 * H * S
            # Y' = W*(4*B-H) - 8*Y^2*S^2
            ry = W * (B << ECC.two[self.FIDX] - H) - \
                 self.P[Y] * self.P[Y] * S2 << ECC.three[self.FIDX]
            rz = S2 * S << ECC.three[self.FIDX]  # Z' =8 * S^3
        else:
            H = W * W - B * ECC.eight[self.FIDX]  # H = W^2 - 8*B
            rx = H * S * ECC.two[self.FIDX]  # X' = 2 * H * S
            # Y' = W*(4*B-H) - 8*Y^2*S^2
            ry = W * (B * ECC.eight[self.FIDX] - H) - \
                 self.P[Y] * self.P[Y] * S2 * ECC.eight[self.FIDX]
            rz = S2 * S * ECC.eight[self.FIDX]  # Z' =8 * S^3

        return ECCProjective([rx, ry, rz])

    # comparison operators
    def __eq__(self, P2):
        """
          P1 == P2
        """
        if not isinstance(P2, ECCProjective):
            assert True, "Unexpected data type"
        elif self.is_zero() and P2.is_zero():
            return True
        else:
            return (self.P[ECC.X] * P2.P[ECC.Z], self.P[ECC.Y] * P2.P[ECC.Z]) == \
                   (P2.P[ECC.X] * self.P[ECC.Z], P2.P[ECC.Y] * self.P[ECC.Z])

    def __ne__(self, P2):
        """
          True if points are different
          False otherwis
        """
        return not (self == P2)


class ECCJacobian(ECC):
    """
      Curve : y^2 = x^3 + a*x + b
      Point on the curve : (x,y) st complies with curve above and
        x,y are part of finite field Fp
    """

    def __init__(self, p, curve=None):
        ECC.__init__(self, p, curve)

    def to_projective(self):
        """
          Convert Jacobian to Projective coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)

          TODO
        """
        pass

    def to_affine(self):
        """
          Converts point to AFFINE
        """
        pass

    def to_jacobian(self):
        """
          Don't do eanything
        """
        pass

    def is_on_curve(self):
        """
          True of point is on curve
          False otherwise
        """
        pass

    # Arithmetic operators
    # +, - , neg, *
    def __add__(self, P2):
        """
          P1 + P2

        """
        pass

    def __sub__(self, P2):
        """
          P1 - P2
          Check P2 is Projective
        """
        pass

    def __neg__(self):
        """
          -P1
        """
        pass

    def __mul__(self, alpha):
        """
          alpha * P1
          TODO : pending += for ECC points and >>= for BigInt, SLINDING window
        """
        pass

    # doubling operation
    def double(self):
        """
         2 * P1
        """
        pass

    # comparison operators
    def __eq__(self, P2):
        """
          P1 == P2
        """
        pass

    def __ne__(self, P2):
        """
          True if points are different
          False otherwis
        """
        pass
