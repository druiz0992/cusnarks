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

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : z2field_element
//
// Date       : 28/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Implementation of quadratic extension finite field functionality and arithmetic. 
//
//       Methods:
//
//   TODO
// ------------------------------------------------------------------

"""
import copy

from zfield import *
from random import randint


class Z2FieldEl(ZFieldEl):
    init = False
    zero = [[None, None], [None, None]]

    def __init__(self, el, force_init=False):
        """
          el : [ZFieldEl, ZFieldEl], Z2FieldEl, None
        """
        if not ZField.is_init():
            assert False, "Finite Field not initialized"
            return

        if Z2FieldEl.init == False or force_init:
            Z2FieldEl.zero = [[ZFieldElExt(0), ZFieldElExt(0)], [ZFieldElRedc(0), ZFieldElRedc(0)]]
            Z2FieldEl.init = True

        if isinstance(el, Z2FieldEl):
            self.P = copy.copy(el.P)

        elif type(el) is list and len(el) == 2 and \
                ((isinstance(el[0], ZFieldElExt) and isinstance(el[1], ZFieldElExt)) or \
                 (isinstance(el[0], ZFieldElRedc) and isinstance(el[1], ZFieldElRedc))):
            self.P = el
        elif type(el) is list and len(el) == 2 and not isinstance(el[0], ZFieldElRedc):
            self.P = [ZFieldElExt(x) for x in el]
        elif el is None:
            self.P = copy.copy(Z2FieldEl.zero[ZUtils.DEFAULT_IN_REP_FORMAT])
        else:
            assert False, "Unexpected data type"

    def __add__(self, other):
        """
          X + Y  
        """
        if (isinstance(self.P[0], ZFieldElExt) and isinstance(self.P[1], ZFieldElExt)) or \
                (isinstance(self.P[0], ZFieldElRedc) and isinstance(self.P[1], ZFieldElRedc)):
            newZ2 = Z2FieldEl(self)
            newZ2.P[0] = newZ2.P[0] + other.P[0]
            newZ2.P[1] = newZ2.P[1] + other.P[1]

            return newZ2
        else:
            assert False, "Invalid type"

    def __iadd__(self, other):
        self = self + other

        return self

    def __sub__(self, other):
        """
          X - Y 
        """
        return self + -other

    def __neg__(self):
        """
         -X (mod P)
        """
        idx = 0
        if isinstance(self.P[0], ZFieldElRedc):
            idx = 1

        newZ2 = Z2FieldEl(self)
        newZ2.P[0] = -newZ2.P[0]
        newZ2.P[1] = -newZ2.P[1]
        return newZ2

    def __iadd__(self, x):
        """
          X = X + Y (mod P) : Add operation is the same for extended and reduced representations.
           Result of addition is of type self. Note that ZFieldElExt and ZFieldElRedc
           cannot be added toguether
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                #newz = (self.bignum + x.bignum)
                newz = self + x
        elif isinstance(x, int) or isinstance(x, int):
            newz = (self.bignum + x)
        else:
            assert False, "Invalid type"

        if isinstance(self.P[0], ZFieldElRedc):
            self = Z2FieldEl(newz).reduce()
        else:
            self = Z2FieldEl(newz)

        return self

    def __isub__(self, x):
        """
          X = X - Y (mod P) : Add operation is the same for extended and reduced representations.
           Result of operation is of type self. Note that ZFieldElExt and ZFieldElRedc
           cannot be substracted toguether
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                newz = self - x
                #newz = (self.bignum - x.bignum)
        elif isinstance(x, int) or isinstance(x, int):
            newz = (self.bignum - x)
        else:
            assert False, "Invalid type"

        if isinstance(self.P[0], ZFieldElRedc):
            self = Z2FieldEl(newz).reduce()
        else:
            self = Z2FieldEl(newz)

        return self

    def square(self):
        nr = ZFieldElExt(ZField.get_extended_p()-1)
        if isinstance(self.P[0], ZFieldElRedc):
            nr= nr.reduce()

        newZ2 = Z2FieldEl(self)
        ab = self.P[0] * self.P[1]
        newZ2.P[1] = ab + ab
        t1 = self.P[0] + nr * self.P[1]
        a2 = self.P[0] + self.P[1]
        newZ2.P[0] = t1 * a2 - (ab + nr * ab)

        return newZ2

    def __mul__(self, other):
        """
          X * Y
        """
        idx = 0

        if isinstance(other, Z2FieldEl):
            if self == other:
                 return self.square()

            elif not ((isinstance(self.P[0], ZFieldElExt) and isinstance(self.P[1], ZFieldElExt)) or \
                    (isinstance(self.P[0], ZFieldElRedc) and isinstance(self.P[1], ZFieldElRedc))):
                assert False, "Invalid type"
      
            nr = ZFieldElExt(ZField.get_extended_p()-1)
            if isinstance(self.P[0], ZFieldElRedc):
              nr= nr.reduce()
              idx = 1

            aA = self.P[0] * other.P[0]
            bB = self.P[1] * other.P[1]

            newZ2 = Z2FieldEl(self)
            newZ2.P[0] = aA + nr * bB
            newZ2.P[1] = ((self.P[0] + self.P[1]) * (other.P[0] + other.P[1])) - aA - bB

            """
            newZ3 = Z2FieldEl(self)
            ab = self.P[0] + self.P[1]
            AB = other.P[0] + other.P[1]
            ab = ab * AB
            newZ3.P[0] = aA - bB
            newZ3.P[1] = aA + bB
            newZ3.P[1] = ab - newZ3.P[1]
            """

            return newZ2

        elif isinstance(other, int) or isinstance(other, int) or isinstance(other, BigInt) or \
                (isinstance(other, ZFieldElRedc) and isinstance(self.P[0], ZFieldElRedc)) or \
                (isinstance(other, ZFieldElExt) and isinstance(self.P[0], ZFieldElExt)):

            scalar = other
            if self.P == Z2FieldEl.zero[idx]:
                return Z2FieldEl(self)

            if scalar == 0:
              return Z2FieldEl(Z2FieldEl.zero[idx])
            elif scalar < 0:
                scalar = -scalar

            newZ2 = Z2FieldEl(Z2FieldEl.zero[idx])
            result = self
            while scalar != 0:
                if scalar & 1 != 0:
                    newZ2 += result
                result = result.double()
                scalar >>= 1
            return newZ2

    def __rmul__(self, alpha):
        return self * alpha

    def __lshift__(self, scalar):
        if isinstance(scalar, int) or isinstance(scalar, int) or isinstance(scalar, BigInt):
            newZ2 = Z2FieldEl(self)
            newZ2.P[0] = newZ2.P[0] << scalar
            newZ2.P[1] = newZ2.P[1] << scalar

            return newZ2


    def double(self):
        """
         X + X
        """
        return self + self

    def reduce(self):
        newZ2 = Z2FieldEl(self)
        newZ2.P[0] = self.P[0].reduce()
        newZ2.P[1] = self.P[1].reduce()

        return newZ2

    def extend(self):
        newZ2 = Z2FieldEl(self)
        newZ2.P[0] = self.P[0].extend()
        newZ2.P[1] = self.P[1].extend()

        return newZ2

    def __eq__(self, other):
        if not ((isinstance(self.P[0], ZFieldElExt) and isinstance(self.P[1], ZFieldElExt)) or \
                (isinstance(self.P[0], ZFieldElRedc) and isinstance(self.P[1], ZFieldElRedc))):
            assert False, "Invalid type"
        if not isinstance(other, Z2FieldEl):
            return False

        return self.P[0] == other.P[0] and self.P[1] == other.P[1]

    def __ne__(self, other):
        return not self == other

    def as_list(self):
        return [self.P[0].as_long(), self.P[1].as_long()]

    """
    def as_long(self):
        return self.as_list()
    """

    def __truediv__(self, x):
        """
         X / Y (mod P) : Defined in child classes
        """
        pass

    def __div__(self, other):
        assert False, "Operation not supported"
        return

    def __pow__(self, other):
        assert False, "Operation not supported"
        return

    def inv(self):
        nr = ZFieldElExt(ZField.get_extended_p()-1)
        if isinstance(self.P[0], ZFieldElRedc):
            nr= nr.reduce()

        t0 = self.P[0] * self.P[0]
        t1 = self.P[1] * self.P[1]
        t2 = t0 - nr * t1
        t3 = t2.inv()

        newZ2 = Z2FieldEl(self)
        newZ2.P[0] = self.P[0] * t3
        newZ2.P[1] = -(self.P[1] * t3)

        return newZ2

    def as_uint256(self):
       x = self.as_list()
       return  [BigInt(x[0]).as_uint256(), BigInt(x[1]).as_uint256()]

    @classmethod
    def rand(cls,n, reduced=False):
      p = ZField.get_extended_p().as_long()
      l = []
      for i in xrange(n):
        x1 = randint(1, p-1)
        x2 = randint(1, p-1)
        el = Z2FieldEl([x1, x2])
        if reduced:
          el = el.reduce()
        l.append(el)

      if n==1:
        return l[0]
      else:
        return l
      
      

