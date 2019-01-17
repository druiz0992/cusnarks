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

from abs import ABCMeta, abstractmethod

class ECC:

   __metaclass__ = ABCMeta

   # coordinate formats
   AFFINE     = 0  
   PROJECTIVE = 1
   JACOBIAN   = 2

   # point coordinate indexes
   X = 0
   Y = 1
   Z = 2
    
   def __init__(self,p, a,b):
       """
         Constructor

         Parameters
         ----------
           p           : list of None, int, long, BigInt or ZFieldEl 
                           of 2 (AFFINE) or 3(JACOBIAN/PROJECTIVE) elements
           a,b         : int/long curve coefficients
       """
       if type(p) is not list or (length(p) < 2 and length(p) != 0) or length(p) > 3:
           assert True, "Invalid point format. Expected a list"
       elif ( (isinstance(self,ECCAffine) and len(p) != 2) or
              (isinstance(self,ECCProjective) and len(p) != 3) or
              (isinstance(self,ECCJacobian) and len(p) != 3) ):
           assert True, "Coordinate format doesn't match with point format"
       elif not ZField.is_init():
           assert True, "Finite field not initialized"
       elif not isinstance(int,a) and not isinstance(long,a):
           assert True, "Unexpected curve parameters"
       elif not isinstance(int,b) and not isinstance(long,b):
           assert True, "Unexpected curve parameters"
       else:
           # p can be a list of int, long, BigInt, ZFieldEl
           if isinstance(p[X],int) or
                isinstance(p[X],long) or 
                isinstance(p[X],BigInt) or
                isinstance(p[X],ZFieldEl):
                   self.P = [ZFieldEl(x) for x in p]
           elif p[X] is None:
              self.P = [None for x in p]

           self.a = a 
           self.b = b

   def __init__(self,P1):
       if not isinstance(ECC,P1):
           assert True, "Invalid Point format"
    
       self.a = P1.a
       self.b = P1.b
       self.P = list(P1.P)

   def get_P(self):
       """
         Returns point P
       """
       return self.P
   
   def get_curve(self):
       """
         Returns curve parameters a,b
       """
       return self.a,self.b
   
   def get_format(self):
       """
         Returns coordinate format (AFFINE/PROJECTIVE/JACOBIAN)
       """
       if isinstance(ECCAffine,self):
           return ECC.AFFINE
       elif isinstance(ECCProjective, self):
           return ECC.PROJECTIVE
       elif isinstance(ECCJacobian, self):
           return ECC.JACOBIAN

   def is_affine(self):
       """
         True if coordinate format AFFINE
         False otherwise
       """
       return self.get_format() == ECC.AFFINE

   def is_projective(self):
       """
         True if coordinate format PROJECTIVE
         False otherwise
       """
       return self.get_format() == ECC.PROJECTIVE

   def is_jacobian(self):
       """
         True if coordinate format JACOBIAN
         False otherwise
       """
       return self.get_format() == ECC.JACOBIAN

   def is_inf(self):
       """
         True if point on infinite
         False otherwise
       """
       return self.P[X] is None

   def __eq__(self, P2):
       """
         True if points are equal
         False otherwise
       """
       if not isinstance(P2,ECC):
           assert True,"Unexpected point format"

       if self.get_format() != P2.get_format():
           return False
       elif self.is_inf() and P2.is_inf():
           return True
       else:
           return (self.a, self.b, self.P) == (P2.a, P2.b, P2.P)

   def __ne__(self, P2):
       """
         True if points are different
         False otherwise
       """
       return not self == P2

   @abstractmethod
   def  __add__(self, P2):
       """
         P1 + P2
       """
       pass

   @abstractmethod
   def __mul__(self, alpha):
       """
         alpha * P1
       """
       pass

   @abstractmethod
   def  double(self) :
       """
        2 * P1
       """
       pass:

   @abstracmethod
   def to_affine(self):
       """
         Converts point to AFFINE
       """
       pass

   @abstractmethod
   def to_projective(self):
       """
         Converts point to PROJECTIVE
       """
       pass

   @abstractmethod
   def to_jacobian(self):
       """
         Converts point to JACOBIAN
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
   def __init__(self,p, a,b):
       super().__init__(p,a,b)

   def __init__(self,P1):
       super().__init__(P1)

   def  __add__(self, P2):
       """
         P1 + P2
         Check P2 is Affine. If it is not affine, change to affine and add
       """
       if not isinstance(ECC,P2):
           return True,"Incorrect point format"

       if not P2.is_affine():
           P2.to_affine()

       if self.is_inf():
           return P2
       elif P2.is_inf():
           return self
       elif self == P2:
           return self.double()
       elif self.P[X] == P2.P[X] and self.P[Y] != P2.P[Y]:
           return ECCAffine([None,None],self.a, self.b)
       else:
           s = (self.P[Y] - P2.P[Y]) * (self.P[X] - P2.P[X]).inv()
           rx = s * s - self.P[X] - P2.P[X]
           ry = s * (self.P[X] - rx) - self.P[Y]

           return ECCAffine([rx,ry],self.a, self.b)


   def __mul__(self, alpha):
       """
         alpha * P1
         TODO : pending += for ECC points and >>= for BigInt, SLINDING window
       """
       if not isinstance(int,alpha) and not isinstance(long,alpha) and
          not isinstance(BigInt,alpha) and not isinstance(ZFieldEl,alpha):
           return True,"Incorrect scalar format"

       scalar = ZFieldEl(alpha)
       newP = ECCAffine([None, None], self.a, self.b)
       temp = self
       while scalar != 0:
           if scalar & 1 != 0:
               result += temp
           temp = temp.double()
           n >>= 1
       return result

   def  double(self) :
       """
        2 * P1
       """
       if self.is_inf() or seld.P[Y]==0:
           return ECCAffine([None, None], self.a, self.b)

       s = self.P[X] * self.P[X] * ZFieldEl(3) + self.a
       s += (self.P[Y] * ZFieldEl(2)).inv()
       rx = s * s - self.P[X] * ZFieldEl(2)
       ry = s * (self.P[X] - rx) - self.P[Y]

       return ECCAffine([rx, ry],self.a, self.b)

   def to_affine(self):
       """
         Converts point to AFFINE
       """
       pass


   def to_projective(self):
       """
         Converts point to PROJECTIVE
       """
       if not self.is_inf():
           newP = [self.P[X], self.P[Y], ZFieldEl(1)] 
           self = ECCProjective(newP,self.a, self.n)

   def to_jacobian(self):
       """
         Converts point to JACOBIAN
       """
       if not self.is_inf():
           newP = [self.P[X], self.P[Y], ZFieldEl(1)] 
           self = ECCJacobian(newP,self.a, self.n)

   def is_on_curve(self):
       """
         True of point is on curve
         False otherwise
       """
       return not self.is_inf() and
            self.P[Y] * self.P[Y] == (self.P[X] * self.P[X] + self.a) * self.P[X] + self.b

class ECCProjective(ECC):
     def __init__(self,p, a,b):
       super().__init__(p,a,b)

   def __init__(self,P1):
       super().__init__(P1)

   def  __add__(self, P2):
       """
         P1 + P2
       """
       pass

   def __mul__(self, alpha):
       """
         alpha * P1
       """
       pass

   def  double(self) :
       """
        2 * P1
       """
       pass:

   def to_affine(self):
       """
         Converts point to AFFINE
       """
       pass

   def to_projective(self):
       """
         Converts point to PROJECTIVE
       """
       pass

   def to_jacobian(self):
       """
         Converts point to JACOBIAN
       """
       pass

   def is_on_curve(self):
       """
         True of point is on curve
         False otherwise
       """
     if self.is_inf():
        return True

    return y**2 - x**3 == b

class ECCJacobian(ECC):
     def __init__(self,p, a,b):
       super().__init__(p,a,b)

   def __init__(self,P1):
       super().__init__(P1)

       def  __add__(self, P2):
       """
         P1 + P2
       """
       pass

   def __mul__(self, alpha):
       """
         alpha * P1
       """
       pass

   def  double(self) :
       """
        2 * P1
       """
       pass:

   def to_affine(self):
       """
         Converts point to AFFINE
       """
       pass

   def to_projective(self):
       """
         Converts point to PROJECTIVE
       """
       pass

   def to_jacobian(self):
       """
         Converts point to JACOBIAN
       """
       pass

   def is_on_curve(self):
       """
         True of point is on curve
         False otherwise
       """
       pass
