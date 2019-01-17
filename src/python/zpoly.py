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

# Montgomery reduction algorithm (Python) based from Project Nayuki 
# Number-theoretic transform and factorization code based in Project Nayuki
# 
# Copyright (c) 2018 Project Nayuki
# All rights reserved. Contact Nayuki for licensing.
# https://www.nayuki.io/page/number-theoretic-transform-integer-dft

# Based some additional code from snipets at
#    https://stackoverflow.com/questions/44770632/fft-division-for-fast-polynomial-division
#
// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : zpoly
//
// Date       : 14/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implements polynomial functionality whose coefficiens below to a finite field
//
// ------------------------------------------------------------------

"""
import math
from zfield import ZField

class ZPoly:
   def __init__(self,degree):
      if not ZField.is_init():
        assert True, "Prime field not initialized"

      self.degree = degree;
      prime = ZField.get_extended()
      self.zcoeff = [ZFieldEl(random.randint(0,prime)) for x in xrange(degree+1)]

   def __init__(self,coeffs):
      if not ZField.is_init():
        assert True, "Prime field not initialized"
      elif type(coeffs) is not list:
        assert True, "Coefficients need to be a list"

      self.degree = degree;
      self.zcoeff = [ZFieldEl(coeffs[x]) for x in xrange(degree+1)]

   def __init__(self,zpoly):
      if not ZField.is_init():
        assert True, "Prime field not initialized"
      elif not isinstance(ZPoly,zpoly):
        assert True, "Argument needs to be of type ZPoly"

      self.degree = zpoly.degree;
      self.zcoeff = [ZFieldEl(zpoly[x]) for x in xrange(degree+1)]
   
    def deg(self):
       return this.degree

    def scale(self, n):
      """
       Multiply polynomial ``p(x)`` with ``x^n``.
        If n is negative, poly ``p(x)`` is divided with ``x^n``, and remainder is
       discarded (truncated division).
      """
      if n >= 0:
          self.zcoeff = self.zcoeff + [ZFieldEl(0)] * m
      else:
          self.zcoeff = self.zcoeff[:n]
      self.degree = len(self.zcoeff) - 1


    def __mul__(self,a):
       """
        Multiply polynomial ``p(x)`` with scalar (constant) ``a``.
       """ 
       if isinstance(int,a) or isinstance(long,a) :
         return ZPoly([p * a for p in self.zcoeff])

       elif isinstance(BigInt,a) :
         return ZPoly([p * a.get() for p in self.zcoeff])

       elif isinstance(ZPoly,a):
         d = self.deg() + a.deg()
         d2 = int(math.ceil(math.log(d,2))
         
         p1 = ZPoly(self)
         p1.extend(d2)
         p2 = ZPoly(a)
         p2.extend(d2)
        
         roots, inv_roots = ZField.get_roots()

         if len(roots) != d2 :
           roots, inv_roots = ZField.find_roots(d2,find_inv_roots=True)
           
         p1.ntt(roots)
         p2.ntt(roots) 

         p1 = ZPoly([p1.zcoeff[i] * p2.zcoeff[i] for i in xrange(d2+1)])
         p1.intt(inv_roots)

         return p1

    def ntt(self, powtable):
      """
       Computes the forward number-theoretic transform of the given vector in place,
       with respect to the given primitive nth root of unity under the given modulus.
       The length of the vector must be a power of 2.
      """
      mod = ZField.get()
      vector = self.zcoeff
      n = len(vector)
      levels = n.bit_length() - 1
      tt=[]
      if 1 << levels != n:
        raise ValueError("Length is not a power of 2")
    
      def reverse(x, bits):
        y = 0
        for i in range(bits):
            y = (y << 1) | (x & 1)
            x >>= 1
        return y

      for i in range(n):
        j = reverse(i, levels)
        if j > i:
            vector[i], vector[j] = vector[j], vector[i]
            tt.append([j,i])

       size = 2
       while size <= n:
          halfsize = size // 2
          tablestep = n // size
          for i in range(0, n, size):
              k = 0
              for j in range(i, i + halfsize):
                  l = j + halfsize
                  left = vector[j]
                  right = long(vector[l]) * long(powtable[k])
                  vector[j] = left + right
                  vector[l] = left - right
                  #print size, i, j, l, hex(mod), hex(powtable[k]), hex(vector[j]), hex(vector[l])
                  k += tablestep
          size *= 2

    def intt(self, powtable):
      """
       Computes the inverse number-theoretic transform of the given vector in place,
       with respect to the given primitive nth root of unity under the given modulus.
       The length of the vector must be a power of 2.
      """
      self.ntt(powtable)
      scaler = ZField.inv(len(powtable), Zfield.get_extended())
      self = self * scaler
 
    def extend(self,d):
      """
       Extend list ``p`` representing a polynomial ``p(x)`` to
        match polynomials of degree ``d-1``.
      """
      self.zcoeff = [ZFieldEl(0)] * (d - len(self.zcoeff)) + self.zcoeff
      self.degree = d

    def __add__(self,v):
      """
        Add polynomials ``u(x)`` and ``v(x)``.
      """
      if not isinstance(ZPoly,v):
        assert True, "Unexpected data type"

      p1 = ZPoly(self)
      p2 = ZPoly(v)

      d = max(p1.deg(), p2.deg())
      p1.extend(d)
      p2.extend(d)

      p1 = ZPoly([p1.zcoeff[i] + p2.zcoeff[i] for i in xrange(d+1)])

      return p1

    def __sub__(self,v):
      """
        Sub polynomials ``u(x)`` and ``v(x)``.
      """
      if not isinstance(ZPoly,v):
        assert True, "Unexpected data type"

      p1 = ZPoly(self)
      p2 = ZPoly(v)

      d = max(p1.deg(), p2.deg())
      p1.extend(d)
      p2.extend(d)

      p1 = ZPoly([p1.zcoeff[i] - p2.zcoeff[i] for i in xrange(d+1)])

      return p1

    def reciprocal(self):
      """
       Calculate the reciprocal of polynomial ``p(x)`` with degree ``k-1``,
       defined as: ``x^(2k-2) / p(x)``, where ``k`` is a power of 2.
      """
     k = self.deg() + 1
     assert k>0 and self.zcoeff[-1] != 0 and 2**round(math.log(k,2)) == k

     if k == 1:
       self = ZPoly([ZFieldEl._inv(self.zcoeff[-1]])

     q = ZPoly(self.zcoeff[:k/2])
     q.reciprocal()
    
     q2 = q * 2
     q2.scale(3*k/2 - 2)
     r = q2 - (q * q * self)
     r.scale(-k+2)

     return r


    def div(self, v):
      """
        Fast polynomial division ``u(x)`` / ``v(x)`` of polynomials with degrees
        m and n. Time complexity is ``O(n*log(n))`` if ``m`` is of the same order
        as ``n``.
      """
      if not isinstance(ZPoly,v):
        assert True, "Unexpected data type"

      m = self.deg()
      n = v.deg()
      
      if m < n
        return ZPoly([0])

      # ensure deg(v) is one less than some power of 2
      # by extending v -> ve, u -> ue (mult by x^nd)
      nd = int(2**math.ceil(math.log(n, 2))) - 1 - n
      ue = ZPoly(self)
      ue.scale( nd)
      ve = ZPoly(v)
      ve.scale(nd)
      me = m + nd
      ne = n + nd

      s = ZPoly(ve)
      s.reciprocal()
   
      q = ue * s
      q = q.scale(-2*ne)

      # handle the case when m>2n
      if me > 2*ne:
          # t = x^2n - s*v
          t = ZPoly([1] * 2*ne) - s * ve
          t = ue * t
          t.scale(-2*ne)
          q2, r2 = t.div(ve)
          q = q + q2

       # remainder, r = u - v*q
       r = u - v * q

      return q, r
