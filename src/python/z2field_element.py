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
from zfield import *


class Z2FieldEl(ZFieldEl):
    init = False
    zero = [[None, None], [None, None]]
    non_residue = [None, None]

    def __init__(self, el, force_init=False):
        """
          el : [ZFieldEl, ZFieldEl], Z2FieldEl
        """
        if not ZField.is_init():
            assert True, "Finite Field not initialized"
            return

        if Z2FieldEl.init == False or force_init:
            zero = [[ZFieldElExt(0),ZFieldElExt(0)], [ZFieldElRedc(0), ZFieldElRedc(0)]]
            Z2FieldEl.non_residue = [ZFieldElExt(ZField.get_extended_p - 1), ZFieldElExt(ZField.get_extended_p - 1).reduce)]
            Z2FieldEl.init = True

        if isinstance(el,Z2FieldEl):
            this.P = el.P

        elif type(el) is list and len(el) == 2 and \
                ((isinstance(el[0],ZFieldElExt) and isinstance(el[1],ZFieldElExt)) or \
                (isinstance(el[0],ZFieldElRedc) and isinstance(el[1],ZFieldElRedc)) ):
            this.P = el
        else :
            assert True, "Unexpected data type"
         
    def __add__(self,other):
        """
          X + Y  
        """
        if  (isinstance(self.P[0],ZFieldElExt) and isinstance(self.P[1],ZFieldElExt)) or \
             (isinstance(self.P[0],ZFieldElRedc) and isinstance(self.P[1],ZFieldElRedc)) :
                newZ2 = Z2FieldEl(self)
                newZ2.P[0] = newZ2.P[0] + other.P[0]
                newZ2.P[1[ = newZ2.P[1] + other.P[1]

                return newZ2
        else:
            assert True, "Invalid type"


    def __iadd__(self,other):
         self = self + other

         return self

    def __sub__(self, other):
        """
          X - Y 
        """
        return self + -other

    def __neg__(self)
        """
         -X (mod P)
        """
        idx=0
        if isinstance(self.P[0], ZFieldElRedc):
           idx = 1

        return Z2FieldEl.zero[idx] - self

    def __mul__(self, other)
        """
          X * Y
        """
        idx=0
        if isinstance(self.P[0], ZFieldElRedc):
           idx = 1

        if isinstance(other, Z2FieldEl):
            if  not (isinstance(self.P[0],ZFieldElExt) and isinstance(self.P[1],ZFieldElExt)) or \
                 (isinstance(self.P[0],ZFieldElRedc) and isinstance(self.P[1],ZFieldElRedc)) :
               assert True, "Invalid type"

            aA = this.P[0] * other.P[0]
            bB = this.P[1] * other.P[1]
            
            newZ2 = Z2FieldEl(self)
            newZ2[0] = aA + Z2FieldEl.non_residue[idx] * bB
            newZ2[1] = ((this.P[0] + this.P[1]) * ( other.P[0] + other.P[1])) - aA + bB
    
            return newZ2
        elif isinstance(other, int) or isinstance(other, long) or isinstance(other,BigInt) or \
                ( isinstance(other,ZFieldElRedc) and isinstance(self.P[0],ZFieldElRedc) ) or \
                ( isinstance(other,ZFieldElExt) and isinstance(self.P[0],ZFieldElExt) ) :

            scalar = other
            if self == Z2FieldEl.zero[idx]
                return Z2FieldEl(self)

            if scalar < 0:
                scalar = -scalar

            newZ2 = Z2FieldEl(Z2FieldEl.zero[idx])
            result = self
            while scalar != 0:
                if scalar & 1 != 0:
                    newP +=  result
                result = result.double()
                scalar >>= 1
            return result

    def __rmul__(self, alpha):
        return self * alpha
    
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

    def __eq__(self,other):
        if  not (isinstance(self.P[0],ZFieldElExt) and isinstance(self.P[1],ZFieldElExt)) or \
             (isinstance(self.P[0],ZFieldElRedc) and isinstance(self.P[1],ZFieldElRedc)) :
           assert True, "Invalid type"

        return self.P[0] == other.P[0] and self.P[1] == other.P[1]

    def __ne__(self, other):
        return not self == other

    def as_list(self):
        return [self.P[0].as_long, self.P[1].as_long()]


