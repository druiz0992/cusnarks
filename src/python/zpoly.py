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
from random import randint

from zfield import *


class ZPoly(object):
    FEXT = 0
    FRDC = 1

    FFT_MUL_THRES = 128


    def __init__(self, p):
        if not ZField.is_init():
            assert True, "Prime field not initialized"
        elif isinstance(p, int) or isinstance(p, long):
            if p < 0:
                assert True, "Polynomial needs to be at least degree 0"
            else:
                self.degree = p;
                prime = ZField.get_extended_p()
                self.zcoeff = [ZFieldElExt(randint(0, prime.as_long() - 1)) for x in xrange(p + 1)]
                self.FIDX = ZPoly.FEXT
        elif type(p) is list or type(p) is dict:
            self.zcoeff, self.degree, self.FIDX = ZPoly.set_properties(p)
        elif isinstance(p, ZPoly):
            self.degree = p.get_degree()
            self.zcoeff = p.get_coeff()
            self.FIDX = p.FIDX
        else:
            assert True, "Unexpected data type"

    @classmethod
    def set_properties(cls, p):
        if type(p) is list:
            degree = len(p) - 1
            if isinstance(p[0], ZFieldElExt):
                zcoeff = p
                FIDX = ZPoly.FEXT
            elif isinstance(p[0], ZFieldElRedc):
                zcoeff = p
                FIDX = ZPoly.FRDC
            elif isinstance(p[0], int) or isinstance(p[0], long) or isinstance(p[0], BigInt):
                zcoeff = [ZFieldElExt(t) for t in p]
                FIDX = ZPoly.FEXT
            else:
                assert True, "Unexpected data type"
        elif type(p) is dict:
            c = sorted([long(k) for k in p.keys()])
            degree = long(c[-1])
            d_k = str(degree)
            if isinstance(p[d_k], ZFieldElExt):
                zcoeff = p
                FIDX = ZPoly.FEXT
            elif isinstance(p[d_k], ZFieldElRedc):
                zcoeff = p
                FIDX = ZPoly.FRDC
            elif isinstance(p[d_k], int) or isinstance(p[d_k], long) or isinstance(p[d_k], BigInt):
                zcoeff = {i: ZFieldElExt(p[i]) for i in p.keys()}
                FIDX = ZPoly.FEXT
            elif isinstance(p, ZPoly):
                zcoeff = p.get_coeff()
                degree = p.get_degree()
                FIDX = p.FIDX
            else:
                assert True, "Unexpected data type"

        return zcoeff, degree, FIDX

    def get_properties(self):
        if type(self.get_coeff()) is list:
            return self.zcoeff, self.degree, self.FIDX
        else:
            return ZPoly.dict_to_list(self.zcoeff), self.degree, self.FIDX

    @classmethod
    def dict_to_list(cls, d):
        c = sorted([long(k) for k in d.keys()])
        if isinstance(d[str(c[0])], ZFieldElRedc):
            l = np.asarray([ZFieldElRedc(0)] * (c[-1] + 1))
        else:
            l = np.asarray([ZFieldElExt(0)] * (c[-1] + 1))

        l[c] = [d[str(k)] for k in c]

        return list(l)

    def to_sparse(self):
        pass

    def to_dense(self):
        pass

    def get_degree(self):
        return self.degree

    def get_coeff(self):
        if type(self.zcoeff) == list:
            return list(self.zcoeff)
        else :
            return dict(self.zcoeff)

    def extend(self):
        if self.FIDX == ZPoly.FRDC:
            return ZPoly([c.extend() for c in self.get_coeff()])
        else:
            return self

    def reduce(self):
        if self.FIDX == ZPoly.FEXT:
            return ZPoly([c.reduce() for c in self.get_coeff()])
        else:
            return self

    def norm(self):
        """
        Normalize poly to have a non zero most significant coefficient
        """
        coeff = self.get_coeff()

        if coeff[-1] != 0:
                return ZPoly(coeff)

        for i,c in enumerate(coeff[::-1]):
            if c != 0:
                return ZPoly(coeff[:-i])

        if self.FIDX == ZPoly.FEXT:
          return ZPoly([ZFieldElExt(0)])
        else:
          return ZPoly([ZFieldElRedc(0)])

    def poly_mul(self, p2):
        """
         Polynomial multiplication (in place). Both polys self and p2 will be modified      
         
         NOTE : Multiplication is done in place

         TODO : Limit poly multiplication to polys with same coefficient format.
            Check later if this can be modified
         TODO : Check if it is worth storing roots in ext and reduced format
         TODO : Force polys are pre-scaled to have a power of 2 number of coeffs
        """
        if not isinstance(p2, ZPoly):    # TODO : For now, limit poly mul to coeffs in same format
            assert True, "Unexpected data type"
        
        if self.FIDX != p2.FIDX:
            assert True, "Coefficients need to be in the same format"
        # TODO : Assuming that poly are not pre-scaled with power of two length. 
        #  enforcing that polys have a power of two number of coefficients will save
        #  this operation

        d1 = self.get_degree()
        d2 = p2.get_degree()

        if d1 == 0 :
            p = self.zcoeff[0] * p2
            self.zcoeff, self.degree, self.FIDX = p.get_properties()

        elif d2 == 0 :
            p = self * p2.zcoeff[0]
            self.zcoeff, self.degree, self.FIDX = p.get_properties()

        elif d1+d2 < ZPoly.FFT_MUL_THRES :
            self.poly_mul_normal(p2)

        elif self == p2:
            self.poly_square()

        else:
            self.poly_mul_fft(p2)

        return       
        
    def poly_mul_normal(self, p2):
         p1 = [c.as_long() for c in self.get_coeff()]
         p2 = [c.as_long() for c in p2.get_coeff()]
         p1 = np.polymul(p1[::-1], p2[::-1])
         if self.FIDX == ZPoly.FEXT:
            self.zcoeff = [ZFieldElExt(c) for c in p1[::-1]]
         else:
            self.zcoeff = [ZFieldElRedc(c) for c in p1[::-1]]

         self.degree = len(self.zcoeff) - 1

         return

    def poly_square(self):
        roots, inv_roots = ZField.get_roots()

        d1 = self.get_degree()
        dtp = 2*d1
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        # Recompute roots in case nroots changed or format.
        if len(roots) != dt or not isinstance(roots[0],type(self.zcoeff[0])):
            roots, inv_roots = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZPoly.FEXT)
        self.expand_to_degree(dt, self)
        self.ntt(roots[:dt/2 + 1])
        for i in xrange(dt+1):
           self.zcoeff[i] = self.zcoeff[i] ** 2

        self.intt(inv_roots[:dt/2 + 1])
        self.expand_to_degree(dtp,self)

    def poly_mul_fft(self, p2): 

        roots, inv_roots = ZField.get_roots()

        d1 = self.get_degree()
        d2 = p2.get_degree()
        dtp = d1 + d2
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        # Recompute roots in case nroots changed or format.
        if len(roots) != dt or not isinstance(roots[0],type(self.zcoeff[0])):
            roots, inv_roots = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZPoly.FEXT)

        self.expand_to_degree(dt, self)
        p2.expand_to_degree(dt, p2)
        self.ntt(roots[:dt/2 + 1])
        p2.ntt(roots[:dt/2 + 1])
        for i in xrange(dt+1):
           self.zcoeff[i] *= p2.zcoeff[i]

        self.intt(inv_roots[:dt/2 + 1])
        self.expand_to_degree(dtp,self)

    def __rmul__(self, a):
        """
         Multiply scalar with polynomial ``p(x)``
        """
        return self * a

    def __mul__(self, a):
        """
         Multiply polynomial ``p(x)`` with scalar (constant) ``a``.
        """
        if isinstance(a, int) or isinstance(a, long) or isinstance(a, BigInt):
            return ZPoly([p * a for p in self.zcoeff])
        else:
            assert True, "Unexpected data type"

    def ntt(self, powtable):
        """
         Computes the forward number-theoretic transform of the given vector in place,
         with respect to the given primitive nth root of unity under the given modulus.
         The length of the vector must be a power of 2.
        """
        vector = self.zcoeff
        n = len(vector)
        levels = n.bit_length() - 1
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

        size = 2
        while size <= n:
            halfsize = size // 2
            tablestep = n // size
            for i in range(0, n, size):
                k = 0
                for j in range(i, i + halfsize):
                    l = j + halfsize
                    left = vector[j]
                    right = vector[l] * powtable[k]
                    vector[j] = left + right
                    vector[l] = left - right
                    # print size, i, j, l, hex(mod), hex(powtable[k]), hex(vector[j]), hex(vector[l])
                    k += tablestep
            size *= 2

    def intt(self, powtable):
        """
         Computes the inverse number-theoretic transform of the given vector in place,
         with respect to the given primitive nth root of unity under the given modulus.
         The length of the vector must be a power of 2.
        """
        self.ntt(powtable)
        nroots = ZFieldElExt(len(powtable)*2)
        if self.FIDX == ZPoly.FEXT:
           scaler = nroots.inv()
        else:
           scaler = nroots.inv().reduce()

        self.zcoeff = [c * scaler for c in self.get_coeff()]

    def expand_to_degree(self, d, ret=None):
        """
         Extend list ``p`` representing a polynomial ``p(x)`` to
          match polynomials of degree ``d-1``.
        """
        if ret is not None:
            newP = ret
        else:
            newP = ZPoly(0)
        
        if d >= self.get_degree() :
            if self.FIDX == ZPoly.FEXT:
                newP.zcoeff = self.get_coeff() + [ZFieldElExt(0)] * (d - self.get_degree())
            else:
                newP.zcoeff = self.get_coeff() + [ZFieldElRedc(0)] * (d - self.get_degree())
                newP.FIDX = ZPoly.FRDC
        else:
            newP.zcoeff = self.get_coeff()[:d+1]

        newP.degree = len(newP.zcoeff)-1

        return newP

    def scale(self, d, ret=None):
        """
         Extend list ``p`` representing a polynomial ``p(x)`` to
          match polynomials of degree ``d-1``.
        """
        if ret is not None:
            newP = ret
        else:
            newP = ZPoly(0)
        
        if d >= 0:
            if self.FIDX == ZPoly.FEXT:
                newP.zcoeff = [ZFieldElExt(0)] * d + self.get_coeff()
            else:
                newP.zcoeff = [ZFieldElRedc(0)] * d + self.get_coeff()
                newP.FIDX = ZPoly.FRDC
        else:
            newP.zcoeff = self.get_coeff()[-d:]

        newP.FIDX =   self.FIDX
        newP.degree = len(newP.zcoeff)-1

        return newP

    def __add__(self, v):
        """
          Add polynomials ``u(x)`` and ``v(x)``.
        """
        if not isinstance(v, ZPoly):
            assert True, "Unexpected data type"

        min_d = min(self.get_degree(), v.get_degree())
        if self.get_degree() >= v.get_degree():
            new_p = ZPoly(self.get_coeff())
        else:
            new_p = ZPoly(v)

        c1 = self.get_coeff()
        c2 = v.get_coeff()

        new_p.zcoeff[:min_d+1] = [c1[i] + c2[i] for i in xrange(min_d + 1)]

        return new_p

    def __sub__(self, v):
        """
          Sub polynomials ``u(x)`` and ``v(x)``.
        """
        return self + (-v)

    def __neg__(self):
        """
          Negate polynomial coeffs
        """
        new_p = ZPoly(self)
        new_p_coeff = new_p.get_coeff()
        new_p.zcoeff = [-c for c in new_p_coeff]

        return new_p

    def zero(self):
        """
          return 0 poly
        """
        if self.FIDX == self.FEXT:
            return ZPoly([ZFieldElExt(0)])
        else:
            return ZPoly([ZFieldElRedc(0)])
    def inv(self):
        """
         Calculate the reciprocal of polynomial ``p(x)`` with degree ``k-1``,
         defined as: ``x^(2k-2) / p(x)``, where ``k`` is a power of 2.
        # TODO : implement double when multiplying by 2
        # TODO : Montgomery inversion doesn't work
        """
        k = self.get_degree() + 1
        assert k > 0 and self.zcoeff[-1] != 0 and 2 ** round(math.log(k, 2)) == k

        if k == 1:
            return ZPoly([self.zcoeff[0].inv()])

        npa = self.scale(-k/2)
        q = npa.inv()
        # TODO : implement double when multiplying by 2. shift, and store 2 as reduced
        if self.FIDX == ZPoly.FEXT:
           a = (q * 2).scale(3*k/2-2)
        else :
           a = ( q * ZFieldElRedc(2)).scale(3*k/2-2)
        q.poly_mul(q)
        b = ZPoly(self)
        b.poly_mul(q)

        r = a - b 
        r.scale(-k + 2, r)

        return r

    def show(self):
       if type(self.get_coeff()) is list:
          print [c.as_long() for c in self.get_coeff()]
       else :
          l =  ZPoly.dict_to_list(self.zcoeff)
          print [c.as_long() for c in l]

    def poly_div(self, v):
        """
          Fast polynomial division ``u(x)`` / ``v(x)`` of polynomials with degrees
          m and n. Time complexity is ``O(n*log(n))`` if ``m`` is of the same order
          as ``n``.
          TODO if degree of u(x) > 2 * degree v(x), function doesn't work!!
          TODO optimize. There are far too many copies in this function and in multiplication
          TODO : Montgomery division doesn't work
          TODO : Two version of division. Keep 1
      
        """
        if not isinstance(v, ZPoly):
            assert True, "Unexpected data type"

        m = self.get_degree()
        n = v.get_degree()

        if m < n:
            return self.zero()

        # ensure deg(v) is one less than some power of 2
        # by extending v -> ve, u -> ue (mult by x^nd)
        nd = (1<<  int(math.ceil(math.log(n+1, 2))) )- 1 - n
        ue = self.scale(nd)
        ve = v.scale(nd)
        me = m + nd
        ne = n + nd

        s = ve.inv()

        # handle the case when m>2n
        if me > 2* ne:
            # t = x^2n - s*v
            ve_copy = ZPoly(ve)
            s_copy = ZPoly(s)
            s_copy.poly_mul(ve_copy)
            t = ZPoly([1]).scale( 2 * ne) - s_copy

        q = self.zero()
        rem = ZPoly(ue)
        done = False

        while not done:
            us = ZPoly(s)
            rem_copy = ZPoly(rem)
            us.poly_mul(rem_copy)
            us = us.scale(-2*ne)
            q = q + us

            if me > 2 * ne:
                t_copy = ZPoly(t)
                rem.poly_mul(t_copy)
                rem = rem.scale(-2*ne)
                #s_copy = ZPoly(s)
                #rem.poly_mul(s_copy)
                #rem.scale(-2*ne)
                me = rem.get_degree()- ne
            else:
                done = True

            rem = ZPoly(us)


        return q

    def poly_div2(self, v):
        """
          Fast polynomial division ``u(x)`` / ``v(x)`` of polynomials with degrees
          m and n. Time complexity is ``O(n*log(n))`` if ``m`` is of the same order
          as ``n``.
        """
        if not isinstance(v, ZPoly):
            assert True, "Unexpected data type"

        m = self.get_degree()
        n = v.get_degree()

        if m < n:
            return self.zero()

        # ensure deg(v) is one less than some power of 2
        # by extending v -> ve, u -> ue (mult by x^nd)
        nd = (1<<  int(math.ceil(math.log(n+1, 2))) )- 1 - n
        ue = self.scale(nd)
        ve = v.scale(nd)
        me = m + nd
        ne = n + nd

        s = ve.inv()

        q = ZPoly(ue)
        s_copy = ZPoly(s)
        q.poly_mul(s_copy)
        q = q.scale(-2 * ne)

        # handle the case when m>2n
        if me > 2 * ne:
            # t = x^2n - s*v
            ve_copy = ZPoly(ve)
            s.poly_mul(ve_copy)
            t = ZPoly([1]).scale( 2 * ne) - s
            t.poly_mul(ue)
            t.scale(-2 * ne,t)
            #q2,r2 = t.poly_div(ve)
            q2 = t.poly_div(ve)
            q = q + q2

            # remainder, r = u - v*q
        #r = ZPoly(v)
        #r.poly_mul(q)
        #r = self - r
        #return q, r

        return q


class ZPolySparse(ZPoly):
    def __init__(self, p):
        ZPoly.__init__(p)

    def extend(self):
        if self.FIDX == ZPoly.FRDC:
            return ZPoly([c.extend() for c in self.get_coeff()])
        else:
            return self

    def reduce(self):
        pass

    def to_sparse(self):
        pass

    def to_dense(self):
        pass

    def scale(self, n):
        """
         Multiply polynomial ``p(x)`` with ``x^n``.
          If n is negative, poly ``p(x)`` is divided with ``x^n``, and remainder is
         discarded (truncated division).
        """
        pass

    def __mul__(self, a):
        """
         Multiply polynomial ``p(x)`` with scalar (constant) ``a``.
        """
        pass

    def norm(self):
        """
        Normalize poly to have a non zero most significant coefficient
        """
        pass

    def __rmul__(self, a):
        """
         Multiply scalar with polynomial ``p(x)``
        """
        pass

    def ntt(self, powtable):
        """
         Computes the forward number-theoretic transform of the given vector in place,
         with respect to the given primitive nth root of unity under the given modulus.
         The length of the vector must be a power of 2.
        """
        pass

    def intt(self, powtable):
        """
         Computes the inverse number-theoretic transform of the given vector in place,
         with respect to the given primitive nth root of unity under the given modulus.
         The length of the vector must be a power of 2.
        """
        pass

    def expand_to_degree(self, d):
        """
         Extend list ``p`` representing a polynomial ``p(x)`` to
          match polynomials of degree ``d-1``.
        """
        pass

    def __add__(self, v):
        """
          Add polynomials ``u(x)`` and ``v(x)``.
        """
        pass

    def __sub__(self, v):
        """
          Sub polynomials ``u(x)`` and ``v(x)``.
        """
        pass

    def __neg__(self):
        """
          Negate polynomial coeffs
        """
        pass

    def reciprocal(self):
        """
         Calculate the reciprocal of polynomial ``p(x)`` with degree ``k-1``,
         defined as: ``x^(2k-2) / p(x)``, where ``k`` is a power of 2.
        """
        pass

    def poly_div(self, v):
        """
          Fast polynomial division ``u(x)`` / ``v(x)`` of polynomials with degrees
          m and n. Time complexity is ``O(n*log(n))`` if ``m`` is of the same order
          as ``n``.
        """
        pass

    def inv(self):
        """
         Calculate the reciprocal of polynomial ``p(x)`` with degree ``k-1``,
         defined as: ``x^(2k-2) / p(x)``, where ``k`` is a power of 2.
        """
        pass
