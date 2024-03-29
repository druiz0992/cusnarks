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

from __future__ import print_function 
from abc import ABCMeta, abstractmethod
from random import randint, sample
from builtins import int 

from zfield import *
from z2field_element import *

import sys

class ECC(object):
    __metaclass__ = ABCMeta

    # point coordinate indexes
    X = 0
    Y = 1
    Z = 2

    constants_init = False

    zero = [None, None]
    one = [None, None]
    two = [None, None]
    three = [None, None]
    four = [None, None]
    eight = [None, None]

    a = [[None, None], [None, None]]
    b = [[None, None], [None, None]]
    Gx = [None, None]
    Gy = [None,None]

    def __init__(self, p, curve=None):
        """
          Constructor

          Parameters
          ----------
            p           : list of int, long, BigInt, ZFieldEl, Z2FieldEl
                            of (AFFINE/JACOBIAN/PROJECTIVE) elements
            curve           : Dictionary with curve coefficients a,b that define curve y^2=x^3+a*x + b. Keys :
              'a' : long/int
              'b' : long/int
        """
        if not ZField.is_init():
            assert False, "Finite field not initialized"
            return

        # Initialize params
        if ECC.constants_init is False:
            ECC.init_constants()

        if curve is None and not ECC.is_curve_init():
            assert False, "Curve is not initialized"
            return
        elif type(p) is list:
            if len(p) == 2 and isinstance(p[ECC.X], ZFieldElRedc):
                  if p[ECC.X] == ECC.zero[ZUtils.FRDC] and p[ECC.Y] == ECC.one[ZUtils.FRDC] : 
                      p = [p[ECC.X], p[ECC.Y], ECC.zero[ZUtils.FRDC]]
                  else:
                      p = [p[ECC.X], p[ECC.Y], ECC.one[ZUtils.FRDC]]
            elif len(p) == 2 and isinstance(p[ECC.X], ZFieldElExt):
                  if p[ECC.X] == ECC.zero[ZUtils.FEXT] and p[ECC.Y] == ECC.one[ZUtils.FEXT] : 
                      p = [p[ECC.X], p[ECC.Y], ECC.zero[ZUtils.FEXT]]
                  else:
                      p = [p[ECC.X], p[ECC.Y], ECC.one[ZUtils.FEXT]]
            elif len(p) == 2 and isinstance(p[ECC.X], Z2FieldEl):
               if isinstance(p[ECC.X].P[0], ZFieldElRedc):
                  if p[ECC.X] == Z2FieldEl([ECC.zero[ZUtils.FRDC], ECC.zero[ZUtils.FRDC]]) and \
                      p[ECC.Y] == Z2FieldEl([ECC.one[ZUtils.FRDC], ECC.zero[ZUtils.FRDC]]):
                    p = [p[ECC.X], p[ECC.Y], Z2FieldEl([ECC.zero[ZUtils.FRDC], ECC.zero[ZUtils.FRDC]])]
                  else:
                    p = [p[ECC.X], p[ECC.Y], Z2FieldEl([ECC.one[ZUtils.FRDC], ECC.zero[ZUtils.FRDC]])]
               else:
                  if p[ECC.X] == Z2FieldEl([ECC.zero[ZUtils.FEXT], ECC.zero[ZUtils.FEXT]]) and \
                      p[ECC.Y] == Z2FieldEl([ECC.one[ZUtils.FEXT], ECC.zero[ZUtils.FEXT]]):
                    p = [p[ECC.X], p[ECC.Y], Z2FieldEl([ECC.zero[ZUtils.FEXT], ECC.zero[ZUtils.FEXT]])]
                  else:
                    p = [p[ECC.X], p[ECC.Y], Z2FieldEl([ECC.one[ZUtils.FEXT], ECC.zero[ZUtils.FEXT]])]
            p_l = p
        elif isinstance(p,ECC):
            p_l = p.P
        else :
            assert False, "Unexpected data type"
            return

        if curve is not None:
           self.init_curve(curve)

        # p can be a list of int, long, BigInt
        if isinstance(p_l[ECC.X], Z2FieldEl) or type(p_l[ECC.X]) is list:
            self.P = [Z2FieldEl(x) for x in p_l]
            self.FIDX = ZUtils.FEXT
            if isinstance(self.P[ECC.X].P[0], ZFieldElRedc):
                self.FIDX = ZUtils.FRDC
        elif isinstance(p_l[ECC.X], int) or isinstance(p_l[ECC.X], int) or isinstance(p_l[ECC.X], ZFieldElExt):
            self.P = [ZFieldElExt(x) for x in p_l]
            self.FIDX = ZUtils.FEXT
        elif isinstance(p_l[ECC.X], ZFieldElRedc):
            self.P = [ZFieldElRedc(x) for x in p_l]
            self.FIDX = ZUtils.FRDC
        else :
            assert False, "Unexpected format"


    @classmethod
    def is_curve_init(cls):
        return ECC.a[ZUtils.FEXT][0] is not None and ECC.b[ZUtils.FEXT][0] is not None

    @classmethod
    def init(cls, curve_params, extended=False):
        if not ZField.is_init():
            assert False, "Field not initialized"
        ECC.init_curve(curve_params, extended=extended)
        ECC.init_constants()

    @classmethod
    def init_constants(cls):
         ECC.zero = [ZFieldElExt(0), ZFieldElRedc(0)]
         ECC.one = [ZFieldElExt(1), ZFieldElExt(1).reduce()]
         ECC.two = [ZFieldElExt(2), ZFieldElExt(2).reduce()]
         ECC.three = [ZFieldElExt(3), ZFieldElExt(3).reduce()]
         ECC.four = [ZFieldElExt(4), ZFieldElExt(4).reduce()]
         ECC.eight = [ZFieldElExt(8), ZFieldElExt(8).reduce()]

         ECC.constants_init = True

    @classmethod
    def init_curve(cls, curve_params, extended=False):
        if 'curve_params' in curve_params:
           cp = curve_params['curve_params']
        else:
           cp = curve_params

        ECC.a[0] = [ZFieldElExt(cp['a']), ZFieldElExt(cp['a']).reduce()]
        ECC.b[0] = [ZFieldElExt(cp['b']), ZFieldElExt(cp['b']).reduce()]
        ECC.Gx[0] = ZFieldElExt(cp['Gx'])
        ECC.Gy[0] = ZFieldElExt(cp['Gy'])

        if 'curve_params_g2' in curve_params:
           cp = curve_params['curve_params_g2']
        elif 'ax1' in curve_params:
           cp = curve_params
        else:
           return

        ECC.a[1] = [Z2FieldEl([cp['ax1'], cp['ax2']]),
                   Z2FieldEl([cp['ax1'], cp['ax2']]).reduce()]
        ECC.b[1] = [Z2FieldEl([cp['bx1'], cp['bx2']]),
                  Z2FieldEl([cp['bx1'], cp['bx2']]).reduce()]
        ECC.Gx[1] = Z2FieldEl([cp['Gx1'], cp['Gx2']])
        ECC.Gy[1] = Z2FieldEl([cp['Gy1'], cp['Gy2']])

    @classmethod
    def p_zero(cls, ext_field=False):

        if ECC.constants_init is False:
            assert False, "Curve not initialized"

        if ext_field is True:
            one = Z2FieldEl([ECC.one[0], ECC.zero[0]])
            zero = Z2FieldEl([ECC.zero[0], ECC.zero[0]])
        else:
            one = ECC.one[0]
            zero = ECC.zero[0]

        inf = [zero, one, zero]
        return  inf

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
        if isinstance(self.P[ECC.X],Z2FieldEl):
            return self.P[ECC.Z].as_list() == [0,0]
        elif isinstance(self.P[ECC.X],ZFieldEl):
            return self.P[ECC.Z].as_long() == 0

    def reduce(self):
        """
         Return new Elliptic curve point with coordinates expressed in Montgomert format
        """
        newP = [x.reduce() for x in self.P]

        if isinstance(self, ECCProjective):
            return ECCProjective(newP)
        elif isinstance(self, ECCJacobian):
            return ECCJacobian(newP)
        elif isinstance(self, ECCAffine):
            return ECCAffine(newP)
        else:
            assert False, "Unexpected data type"

    def extend(self):
        """
         Return new Elliptic curve point with coordinates expressed in extended format
        """
        newP = [x.extend() for x in self.P]

        if isinstance(self, ECCProjective):
            return ECCProjective(newP)
        elif isinstance(self, ECCJacobian):
            return ECCJacobian(newP)
        elif isinstance(self, ECCAffine):
            return ECCAffine(newP)
        else:
            assert False, "Unexpected data type"

    def as_list(self):
        if isinstance(self.P[0] ,Z2FieldEl):
            return [[p.P[0].as_long(), p.P[1].as_long()] for p in self.P]
        else:
            return [p.as_long() for p in self.P]

    def as_str(self):
        if isinstance(self.P[0] ,Z2FieldEl):
            return [[str(p.P[0].as_long()), str(p.P[1].as_long())] for p in self.P]
        else:
            return [str(p.as_long()) for p in self.P]

    def showP(self):
        print(self.as_list())

    def same_format(self, P2):
        """
         returns True of self and P2 are same format of ECC coordinate representation
        """
        if isinstance(P2,Z2FieldEl):
            return type(self.P[0][0]) == type(P2.P[0][0])
        else:
            return type(self.P[0]) == type(P2.P[0])

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
        if isinstance(alpha, int) or isinstance(alpha, int):
            scalar = alpha
        elif isinstance(alpha, ZFieldElRedc):
            assert False, "Unexpected type"
        elif isinstance(alpha,BigInt):
            scalar = alpha.bignum
        else:
            assert False, "Unexpected type"

        if self.is_inf():
            return self.point_at_inf()
        elif scalar == 0:
            return self.point_at_inf()

        if scalar < 0:
            scalar = -scalar

        newP = self.point_at_inf()
        result = self

        for idx in xrange(256):
            b0 = (scalar & (1 << (255-idx))) >> (255 - idx)
            newP = newP.double()
            if b0 != 0:
                newP = result + newP
        return newP

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

    @staticmethod
    def as_uint256(P, remove_last=False, as_reduced=False):
      if remove_last:
         last_idx=2
      else:
         last_idx = 3
      if isinstance(P,list) or isinstance(P,np.ndarray) and isinstance(P[0],ECC):
         if as_reduced:
            Px = [x.reduce().get_P() for x in P]
         else :
            Px = [x.get_P() for x in P]
         Pc=[]
         for Py in Px:
           Pc.append([ x.as_uint256() for x in Py[0:last_idx]])
         Pc = np.concatenate(Pc)
      else:
         if as_reduced:
           Pc = np.asarray([ x.as_uint256() for x in P.reduce().get_P()[0:last_idx]])
         else:
           Pc = np.asarray([ x.as_uint256() for x in P.get_P()[0:last_idx]])

      if len(Pc[0])==2:
          Pc = np.reshape(Pc,(-1,8))

      return Pc

    @staticmethod
    def from_uint256(x, in_ectype=0, out_ectype=0,reduced=False,ec2=False, remove_last = False):
        """

        :param x:
        :return:
        """
        if remove_last:
         last_idx=2
        else:
         last_idx = 3
        
        if not ec2:
          if reduced:
              P = np.reshape(np.asarray([ZFieldElRedc(BigInt.from_uint256(x_).as_long()) for x_ in x]),(-1,last_idx))
          else:
              P = np.reshape(np.asarray([ZFieldElExt(BigInt.from_uint256(x_).as_long()) for x_ in x]),(-1,last_idx))
        else:
          if reduced:
              P = np.reshape(np.asarray([Z2FieldEl([ZFieldElRedc(BigInt.from_uint256(x_[0])), ZFieldElRedc(BigInt.from_uint256(x_[1]))]) for x_ in x]),(-1,last_idx))
          else:
              P = np.reshape(np.asarray([Z2FieldEl([BigInt.from_uint256(x_[0]).as_long(), BigInt.from_uint256(x_[1]).as_long()]) for x_ in x]),(-1,last_idx))
        

        Pinf = ECC._point_at_inf(out_ectype=out_ectype, reduced=reduced, ec2=ec2).get_P()

        for idx,p in enumerate(P):
          if all(p[:2] == Pinf[:2]):
            if remove_last:
              P[idx] = list(Pinf[:2])
            else:
              P[idx] = list(Pinf)

        if in_ectype == 0:
            P_ = [ECCProjective(x_.tolist()) for x_ in P]
        elif in_ectype == 1:
            P_ = [ECCJacobian(x_.tolist()) for x_ in P]
        elif in_ectype == 2:
            P_ = [ECCAffine(x_.tolist()) for x_ in P]
        else:
            assert False, "Unexpected type"

        if in_ectype != out_ectype:
            if out_ectype == 0:
                P_ = [x_.to_projective() for x_ in P_]
            elif out_ectype == 1:
                P_ = [x_.to_jacobian() for x_ in P_]
            elif out_ectype == 2:
                P_ = [x_.to_affine() for x_ in P_]
            else:
                assert False, "Unexpected type"

        return P_

    @staticmethod
    def rand(n,ectype=0, reduce=False, ec2=False, verbose=""):
      """
       generate random point on curve
         n : n random points
         ectype = 0 -> projective, 1 -> jacobian, 2 -> affine
      """
      p = ZField.get_extended_p().as_long()
      ct = 0
      P = []
      P_rdc = []
      for i in xrange(n):
         k = randint(1,p-1)  # generate random number between 1 and p-1

         #P1 = ECCJacobian([ECC.Gx,ECC.Gy, 1])
         if ec2:
             P1 = ECCJacobian([ECC.Gx[1],ECC.Gy[1]])
         else :
             P1 = ECCJacobian([ECC.Gx[0],ECC.Gy[0]])
         P1 = k * P1
         P1 = P1.to_affine()

         
         if ectype == 0:
             P1 = P1.to_projective()
         elif ectype == 1:
             P1 = P1.to_jacobian()
         elif ectype == 2:
              P1 = P1.to_affine()
         else :
              assert False, "Unexpected type"

         if  reduce:
             P_rdc.append(P1.reduce())

         P.append(P1)
         if verbose is not None:
             if ct%10 == 0:
                print(verbose+str(ct)+"\r",end='')
                sys.stdout.flush()
             ct+=1

      if verbose is not None:
          print("\n",end='')

      return P, P_rdc

    @abstractmethod
    def is_on_curve(self):
        """
          True of point is on curve
          False otherwise
        """
        pass

    @staticmethod
    def _point_at_inf(reduced=False, ec2=False, out_ectype=2):
        """"
          Return point at infinity
        """
        idx=0
        if reduced:
           idx= 1
        if ec2:
            one = Z2FieldEl([ECC.one[idx], ECC.zero[idx]])
            zero = Z2FieldEl([ECC.zero[idx], ECC.zero[idx]])
        else:
            one = ECC.one[idx]
            zero = ECC.zero[idx]

        inf = [zero, one, zero]
        if out_ectype==0 :
            newP =  ECCProjective(inf)
        elif out_ectype == 1:
            newP =  ECCAffine(inf)
        else:
            newP =  ECCJacobian(inf)

        return newP

    def point_at_inf(self):
        """"
          Return point at infinity
        """
        if isinstance(self.P[0],Z2FieldEl):
            one = Z2FieldEl([ECC.one[self.FIDX], ECC.zero[self.FIDX]])
            zero = Z2FieldEl([ECC.zero[self.FIDX], ECC.zero[self.FIDX]])
        else:
            one = ECC.one[self.FIDX]
            zero = ECC.zero[self.FIDX]

        inf = [zero, one, zero]
        if isinstance(self,ECCAffine):
            newP =  ECCAffine(inf)
        elif isinstance(self,ECCProjective):
            newP =  ECCProjective(inf)
        else:
            newP =  ECCJacobian(inf)

        if self.FIDX is None:
            newP.FIDX = ZUtils.DEFAULT_IN_PFORMAT
        else:
            newP.FIDX = self.FIDX


        return newP

class ECCAffine(ECC):
    """
      Curve : y^2 = x^3 + a*x + b
      Point on the curve : (x,y) st complies with curve above and
        x,y are part of finite field Fp
    """

    def __init__(self, p, curve=None,force_init=False):
        if force_init :
            ECC.constants_init = False
        ECC.__init__(self, p, curve=curve)

    def to_projective(self):
        """
          Convert Affine to Projective coordinates. Point coordinates are maintained in their
           actual format (Montgomery/Extended)
        """
        if self.is_inf():
            return ECCProjective(self)
        elif isinstance(self.P[ECC.X], Z2FieldEl):
            return ECCProjective([self.P[ECC.X], self.P[ECC.Y], Z2FieldEl([ECC.one[self.FIDX], ECC.zero[self.FIDX]])])
        elif isinstance(self.P[ECC.X], ZFieldEl):
            return ECCProjective([self.P[ECC.X], self.P[ECC.Y], ECC.one[self.FIDX]])
        else:
            assert False, "Unexpected data type"

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
            return ECCJacobian(self)
        elif isinstance(self.P[ECC.X], Z2FieldEl):
            return ECCJacobian([self.P[ECC.X], self.P[ECC.Y], Z2FieldEl([ECC.one[self.FIDX], ECC.zero[self.FIDX]])])
        elif isinstance(self.P[ECC.X], ZFieldEl):
            return ECCJacobian([self.P[ECC.X], self.P[ECC.Y], ECC.one[self.FIDX]])
        else:
            assert False, "Unexpected data type"

    def is_on_curve(self):
        """
          True of point is on curve
          False otherwise
        """
        idx=0
        if isinstance(self.P[0],Z2FieldEl):
          idx=1
        return not self.is_inf() and \
               (self.P[ECC.Y] * self.P[ECC.Y]) == \
               (self.P[ECC.X] * self.P[ECC.X] * self.P[ECC.X]) + \
               (ECC.a[idx][self.FIDX] * self.P[ECC.X]) + ECC.b[idx][self.FIDX]
 
      
    # Arithmetic operators
    # +, - , neg, *
    def __add__(self, P2):
        """
          P1 + P2

        """
        if not isinstance(P2,ECCAffine):
            assert False, "Incorrect point format"
        elif not self.same_format(P2):
            assert False, "Finite Field represenation does not match"

        if self.is_inf():
            return ECCAffine(P2)
        elif P2.is_inf():
            return ECCAffine(self)
        elif self == P2:
            return self.double()
        elif self.P[ECC.X] == P2.P[ECC.X] and self.P[ECC.Y] != P2.P[ECC.Y]:
            return self.point_at_inf()
        else:
            s = (self.P[ECC.Y] - P2.P[ECC.Y]) * (self.P[ECC.X] - P2.P[ECC.X]).inv()
            rx = s * s - self.P[ECC.X] - P2.P[ECC.X]
            ry = s * (self.P[ECC.X] - rx) - self.P[ECC.Y]

        if isinstance(self.P[0],Z2FieldEl):
           one = Z2FieldEl([ECC.one[self.FIDX], ECC.zero[self.FIDX]])
        else:
           one = ECC.one[self.FIDX]

        return ECCAffine([rx, ry, one])

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
            return ECCAffine(self)
        else:
            if isinstance(self.P[0],Z2FieldEl):
                one = Z2FieldEl([ECC.one[self.FIDX], ECC.zero[self.FIDX]])
            else:
                one = ECC.one[self.FIDX]
            return ECCAffine([self.P[ECC.X], -self.P[ECC.Y], one])

    # doubling operation
    def double(self):
        """
         2 * P1
         https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Affine_Coordinates
         1LD + 2M + 2S
        """
        #if self.is_inf() or self.P[ECC.Y].as_list() == [0] or self.P[ECC.Y].as_list() == [0,0]:
        if self.is_inf(): 
            return self.point_at_inf()

        X = self.P[ECC.X]
        Xsq = X * X
        Y = self.P[ECC.Y]

        if isinstance(self.P[0],Z2FieldEl):
          a = ECC.a[1][self.FIDX]
        else:
          a = ECC.a[0][self.FIDX]

        Y2 = Y + Y
        X2 = X + X
        l = Xsq + Xsq + Xsq

        l += a
        l = l * Y2.inv()
        rx = l * l - X2
        ry = l *(X - rx) - Y

        if isinstance(self.P[0],Z2FieldEl):
            onef = Z2FieldEl([ECC.one[self.FIDX], ECC.zero[self.FIDX]])
        else:
            onef = ECC.one[self.FIDX]

        return ECCAffine([rx, ry, onef])

        # comparison operators

    def __eq__(self, P2):
        """
          P1 == P2
        """
        if self.is_inf() and P2.is_inf():
          return True
        elif not isinstance(P2, ECCAffine):
            assert False, "Unexpected data type"
        else:
            return (self.P[ECC.X], self.P[ECC.Y]) == (P2.P[ECC.X], P2.P[ECC.Y])

class ECCProjective(ECC):
    """
      Curve : y^2 = x^3 + a*x + b
      Point on the curve : (x,y) st complies with curve above and
        x,y are part of finite field Fp
    """

    def __init__(self, p, curve=None,force_init=False):
        if force_init :
           ECC.constants_init = False

        ECC.__init__(self, p, curve=curve)

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
        if isinstance(self.P[0],Z2FieldEl):
            one = Z2FieldEl([ECC.one[self.FIDX], ECC.zero[self.FIDX]])
        else:
            one = ECC.one[self.FIDX]

        if self.is_inf():
            return ECCAffine(self)
        elif isinstance(self.P[ECC.X], ZFieldEl):
            div = self.P[ECC.Z].inv()
            return ECCAffine([self.P[ECC.X] * div, self.P[ECC.Y] * div, one])
        else:
            assert False, "Unexpected data type"

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
        idx=0
        if isinstance(self.P[0],Z2FieldEl):
          idx=1
        return not self.is_inf() and \
               self.P[ECC.Y] * self.P[ECC.Y] * self.P[ECC.Z] == self.P[ECC.X] * self.P[ECC.X] * self.P[ECC.X] + \
               ECC.a[idx][self.FIDX] * self.P[ECC.X] * self.P[ECC.Z] * self.P[ECC.Z] + \
               ECC.b[idx][self.FIDX] * self.P[ECC.Z] * self.P[ECC.Z] * self.P[ECC.Z]

    # Arithmetic operators
    # +, - , neg, *
    def __add__(self, P2):
        """
          P1 + P2
        https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Standard_Projective_Coordinates
        12M + 2S
        """
        if not isinstance(P2,ECCProjective):
            assert False, "Incorrect point format"
        elif not self.same_format(P2):
            assert False, "Finite Field represenation does not match"

        if self.is_inf():
            return ECCProjective(P2)
        elif P2.is_inf():
            return ECCProjective(self)

        X1, Y1, Z1 = self.P
        X2, Y2, Z2 = P2.P

        U1 = Y2 * Z1
        U2 = Y1 * Z2
        V1 = X2 * Z1
        V2 = X1 * Z2

        if V1 == V2:
          if U1 != U2:
                return self.point_at_inf(self)
          else:
                return self.double()

        U = U1 - U2
        V = V1 - V2
        Usq = U*U
        Vsq = V*V
        Vcube = Vsq * V
        W = Z1 * Z2
        A = (Usq * W) - Vcube
      
        A -= (Vsq * V2) 
        A -= (Vsq * V2) 

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
            return ECCProjective(self)
        else:
            return ECCProjective([self.P[ECC.X], -self.P[ECC.Y], self.P[ECC.Z]])

    def double(self):
        """
         2 * P1
        https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Standard_Projective_Coordinates
        7M + 5S / 7M + 3S
        """
        #if self.is_inf() or self.P[ECC.Y].as_list() == [0] or self.P[ECC.Y].as_list() == [0,0]:
        if self.is_inf():
            return ECCProjective(self)

        X,Y,Z = self.P

        if isinstance(self.P[0],Z2FieldEl):
           a = ECC.a[1][self.FIDX]
           one = Z2FieldEl([ECC.one[self.FIDX], ECC.zero[self.FIDX]])
           two = Z2FieldEl([ECC.two[self.FIDX], ECC.zero[self.FIDX]])
           three = Z2FieldEl([ECC.three[self.FIDX], ECC.zero[self.FIDX]])
           four = Z2FieldEl([ECC.four[self.FIDX], ECC.zero[self.FIDX]])
           eight = Z2FieldEl([ECC.eight[self.FIDX], ECC.zero[self.FIDX]])
        else:
           a = ECC.a[0][self.FIDX]
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
            W = W + W + W
        else:  # W = a*Z^2 + 3*X^2
            W = a * Z * Z
            Xsq = X * X
            W1 = Xsq + Xsq + Xsq
            W += W1

        H = W * W - eight*B  # H = W^2 - 8*B
        rx = H * (S + S )   # X' = 2 * H * S
        # Y' = W*(4*B-H) - 8*Y^2*S^2
        #ry = W * (B * four - H) - Ysq * Ssq * eight
        ry = W * (four*B  - H) - Ysq * Ssq * eight
        #rz = Scube * eight  # Z' =8 * S^3
        rz = eight * Scube

        return ECCProjective([rx, ry, rz])

    # comparison operators
    def __eq__(self, P2):
        """
          P1 == P2
        """
        if self.is_inf() and P2.is_inf():
            return True
        elif not isinstance(P2, ECCProjective):
            assert False, "Unexpected data type"
        else:
            return (self.P[ECC.X] * P2.P[ECC.Z], self.P[ECC.Y] * P2.P[ECC.Z]) == \
                   (P2.P[ECC.X] * self.P[ECC.Z], P2.P[ECC.Y] * self.P[ECC.Z])

class ECCJacobian(ECC):
    """
      Curve : y^2 = x^3 + a*x + b
      Point on the curve : (x,y) st complies with curve above and
        x,y are part of finite field Fp
    """

    def __init__(self, p, curve=None,force_init=False):
        if force_init :
             ECC.constants_init = False
        ECC.__init__(self, p, curve=curve)

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
        if isinstance(self.P[0],Z2FieldEl):
            one = Z2FieldEl([ECC.one[self.FIDX], ECC.zero[self.FIDX]])
        else:
            one = ECC.one[self.FIDX]

        if self.is_inf():
            return ECCAffine(self)
        elif isinstance(self.P[ECC.X], ZFieldEl):
            zinv = self.P[ECC.Z].inv()
            zinv_sq = zinv * zinv
            zinv_cube = zinv_sq * zinv
            return ECCAffine([self.P[ECC.X] * zinv_sq, self.P[ECC.Y] * zinv_cube, one])
        else:
            assert False, "Unexpected data type"

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
            assert False, "Incorrect point format"
        elif not self.same_format(P2):
            assert False, "Finite Field represenation does not match"

        if self.is_inf():
            return ECCJacobian(P2)
        elif P2.is_inf():
            return ECCJacobian(self)

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

        X3 = (R * R) - Hcube
        X3 = X3 - (U1 * (Hsq + Hsq) )

        Y3 = R * (U1*Hsq - X3) - (S1 * Hcube)
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
            return ECCJacobian(self)
        else:
            return ECCJacobian([self.P[ECC.X], -self.P[ECC.Y], self.P[ECC.Z]])

    # doubling operation
    def double(self):
        """
         2 * P1

         4M + 6S / 4M + 4S
        """
        #if self.is_inf() or self.P[ECC.Y].as_list() == [0] or self.P[ECC.Y].as_list() == [0,0]:
        if self.is_inf():
            return self.point_at_inf()

        X,Y,Z = self.get_P()

        Ysq = Y * Y
        Ysqsq = Ysq * Ysq
        Zsq = Z * Z

        S = X * Ysq
        S = (S + S + S + S)

        if isinstance(self.P[0],Z2FieldEl):
           a = ECC.a[1][self.FIDX]
        else:
           a = ECC.a[0][self.FIDX]

        if a == -ECC.three[self.FIDX]:
            M1 = (X + Zsq) * (X - Zsq)
            M  = (M1 + M1 + M1)
        else:
            M1 = X * X
            M = (M1 + M1 + M1)
            M = M + a * Zsq * Zsq

        X3 = M * M

        Z3 = Y * Z
        X3 = X3 - (S + S)
        t = Ysqsq + Ysqsq
        t = t + t
        t = t + t
        Y3 = M * (S - X3) - t
        Z3 = Z3 + Z3 

        return ECCJacobian([X3, Y3, Z3])

    # comparison operators
    def __eq__(self, P2):
        """
          P1 == P2
        """
        if self.is_inf() and P2.is_inf():
            return True
        elif not isinstance(P2, ECCJacobian):
            assert False, "Unexpected data type"
        else:
            Z1sq = self.P[ECC.Z] * self.P[ECC.Z]
            Z1cube = Z1sq * self.P[ECC.Z]
            Z2sq = P2.P[ECC.Z] * P2.P[ECC.Z]
            Z2cube = Z2sq * P2.P[ECC.Z]

            return (self.P[ECC.X] * Z2sq, self.P[ECC.Y] * Z2cube) == \
                   (P2.P[ECC.X] * Z1sq, P2.P[ECC.Y] * Z1cube)

def ECC_F1(p=None):
    if p is None:
        p = ECC.p_zero(ext_field=False)
    if ZUtils.DEFAULT_IN_REP_FORMAT == ZUtils.AFFINE:
        return ECCAffine(p)
    elif ZUtils.DEFAULT_IN_REP_FORMAT == ZUtils.JACOBIAN:
        return ECCJacobian(p)
    elif ZUtils.DEFAULT_IN_REP_FORMAT == ZUtils.PROJECTIVE:
        return ECCProjective(p)
    else :
        assert False, "Unexpected type"


def ECC_F2(p=None):
    if p is None:
        p = ECC.p_zero(ext_field=True)
    if ZUtils.DEFAULT_IN_REP_FORMAT == ZUtils.AFFINE:
        return ECCAffine(p)
    elif ZUtils.DEFAULT_IN_REP_FORMAT == ZUtils.JACOBIAN:
        return ECCJacobian(p)
    elif ZUtils.DEFAULT_IN_REP_FORMAT == ZUtils.PROJECTIVE:
        return ECCProjective(p)
    else :
        assert False, "Unexpected type"
