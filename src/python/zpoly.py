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

# Number-theoretic transform code based in Project Nayuki
# 
# Copyright (c) 2018 Project Nayuki
# All rights reserved. Contact Nayuki for licensing.
# https://www.nayuki.io/page/number-theoretic-transform-integer-dft

# Based some additional code from snipets at
#    https://stackoverflow.com/questions/44770632/fft-division-for-fast-polynomial-division
#
# Based some additional code from snipets at
#           https://github.com/iden3/iden3js
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
//  Implements polynomial functionality whose coefficiens below to a finite field. A 
//  polynomial p(x) is defined as a0 + a1 * X + a2 * x^2 + ... + an_1 * X^n-1a.
//
//  There are two poly representations defined in two different classes
//   Dense -> class ZPoly(object)        : Poly with all coeffs defined in a list
//   Sparse -> class ZPolySparse(ZPoly)  : Poly with coeffs defined in dictionary. 
//
//  TODO : Check if i need reverse operation in ntt. Maybe I can live without reversing in NTT/INTT
//  TODO : After every operation (poly mul. scalar mul, add , sub, << include normalization to remove unwanted
//            coeffs
// ------------------------------------------------------------------

"""
import math
from random import randint

from zfield import *


class ZPoly(object):
    # beyond this degree, poly mul is done with FFT
    FFT_MUL_THRES = 128

    one = [None, None]
    two = [None, None]

    init = False


    def __init__(self, p, force_init=False):
        """
          ZPoly Constructor. A ZPoly p(x) is defined as p(x) = a0 + a1*x + a2*x^2 + ... + an_1 * x^n_1.
             Polynomial coefficients a0, a1,.., an_1 belong to a pre defined Finite Field. Polynomial member variables are:
            - degree : Polynomial degree. 
            - zcoeff : Coefficients stored in a list as [a0, a1,.., an-1] or in a dic with coeffs as {'0' : a0, '1': a1,..,'n_1':an_1}
            - FIDX   : coefficients format. ZPoly FEXT -> default / ZPoly/FRDC -> Montgomery reduced

          NOTE : Requires ZField to be initialized

          Arguments:
          ------------
           Single input argument with different meanings depending on type.
           
           - int/long/BigInt (Extended) : Polynomial degree. It needs to be positive. A poly of these degree is initialized with
               randomp coefficients. Highest degree coefficient is always nonzero. 

           - list of int/long/BigInt/ZFieldEl : Array with coefficients.

           - dictionary : Keys are strings with coefficients as string. Values are int/long/BigInt/ZFieldEl indicating coefficients.
              ext: p = {'1' : 45, '23':54} creates poly p(x) = 45*x + 54*x^23

           - ZPoly : New Poly is copied
        """
        if force_init:
            ZPoly.init = False

        if not ZField.is_init():
            assert True, "Prime field not initialized"
        elif isinstance(p, int) or isinstance(p, long) or isinstance(p, BigInt):
            if p < 0:
                assert True, "Polynomial needs to be at least degree 0"
            else:
                self.degree = p;
                prime = ZField.get_extended_p()
                self.zcoeff = [ZFieldElExt(randint(0, prime.as_long() - 1)) for x in xrange(p + 1)]
                self.zcoeff[-1] = ZFieldElExt(randint(1, prime.as_long() - 1))
                self.FIDX = ZUtils.FEXT
        elif type(p) is list and isinstance(self, ZPolySparse) :
            assert True, "Unexpected data type"
        #elif type(p) is list or type(p) is dict:
        elif type(p) is list or isinstance(self, ZPolySparse):
            self.zcoeff, self.degree, self.FIDX = ZPoly.set_properties(p)
        elif isinstance(p, ZPoly):
            self.degree = p.get_degree()
            self.zcoeff = p.get_coeff()
            self.FIDX = p.FIDX
        else:
            assert True, "Unexpected data type"

        if ZPoly.init == False:
            ZPoly.one = [ZFieldElExt(1), ZFieldElExt(1).reduce()]
            ZPoly.two = [ZFieldElExt(2), ZFieldElExt(2).reduce()]
            ZPoly.init = True

    @classmethod
    def set_properties(cls, p):
        """
         Retrieve poly properties coeff, degree and FIDX from list/dictionaty with coefficients.

         p : List/Dict with coefficients

        return zcoeff, degree, FIDX
        """
        if type(p) is list:
            degree = len(p) - 1
            if isinstance(p[0], ZFieldElExt):
                zcoeff = p
                FIDX = ZUtils.FEXT
            elif isinstance(p[0], ZFieldElRedc):
                zcoeff = p
                FIDX = ZUtils.FRDC
            elif isinstance(p[0], int) or isinstance(p[0], long) or isinstance(p[0], BigInt):
                zcoeff = [ZFieldElExt(t) for t in p]
                FIDX = ZUtils.FEXT
            else:
                assert True, "Unexpected data type"
        elif type(p) is dict:
            c = sorted([long(k) for k in p.keys()])
            degree = long(c[-1])
            d_k = str(degree)
            if isinstance(p[d_k], ZFieldElExt):
                zcoeff = p
                FIDX = ZUtils.FEXT
            elif isinstance(p[d_k], ZFieldElRedc):
                zcoeff = p
                FIDX = ZUtils.FRDC
            elif isinstance(p[d_k], int) or isinstance(p[d_k], long) or isinstance(p[d_k], BigInt):
                zcoeff = {i: ZFieldElExt(p[i]) for i in p.keys()}
                FIDX = ZUtils.FEXT
            elif isinstance(p, ZPoly):
                zcoeff = p.get_coeff()
                degree = p.get_degree()
                FIDX = p.FIDX
            else:
                assert True, "Unexpected data type"

        return zcoeff, degree, FIDX

    def get_properties(self):
        """
          Returns poly properties zcoeff (as list), degree, FIDX
        """
        if type(self.get_coeff()) is list:
            return self.zcoeff, self.degree, self.FIDX
        else:
            return self.dict_to_list(), self.degree, self.FIDX

    def dict_to_list(self):
        """
          Returns dictionary of coeffs to list of coeffs.
        """

        c = sorted([long(k) for k in self.zcoeff.keys()])
        l = np.asarray(self.zero().get_coeff() * (c[-1] + 1))
        l[c] = [self.zcoeff[str(k)] for k in c]

        return list(l)

    def to_dense(self):
        """
          Do nothing
        """
        return self

    def to_sparse(self):
        """
          Return sparse representation of dense poly. 

          TODO : implement to_sparse
        """
        assert True, "Operation not supported"


    def get_degree(self):
        """
          Get poly degree
        """
        return self.degree

    def get_coeff(self):
        """
          Get poly coeffs as list or dict
        """
        if type(self.zcoeff) == list:
            return list(self.zcoeff)
        else :
            return dict(self.zcoeff)

    def extend(self):
        """
          Convert poly coeffs from REDC to EXT 
        """
        if self.FIDX == ZUtils.FRDC:
            return ZPoly([c.extend() for c in self.get_coeff()])
        else:
            return self

    def reduce(self):
        """
          Convert poly coeffs from EXT to REDC
        """
        if self.FIDX == ZUtils.FEXT:
            return ZPoly([c.reduce() for c in self.get_coeff()])
        else:
            return self

    def norm(self):
        """
        Normalize poly to have a non zero most significant coefficient

         NOTE   https://stackoverflow.com/questions/44770632/fft-division-for-fast-polynomial-division
        """
        coeff = self.get_coeff()

        if coeff[-1] != 0:
                return ZPoly(coeff)

        for i,c in enumerate(coeff[::-1]):
            if c != 0:
                return ZPoly(coeff[:-i])

        return self.zero()

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
         """
          Normal poly implementation O(n^2)

         """
         if self.FIDX == ZUtils.FEXT:
             pa = self
             pb = p2
         else:
             pa = self.extend()
             pb = p2.extend()

         p1 = [c.as_long() for c in pa.get_coeff()]
         p2 = [c.as_long() for c in pb.get_coeff()]
         p1 = np.polymul(p1, p2)

         if self.FIDX == ZUtils.FEXT:
            self.zcoeff = [ZFieldElExt(c) for c in p1]
         else:
            self.zcoeff = [ZFieldElExt(c).reduce() for c in p1]

         self.degree = len(self.zcoeff) - 1

         return

    def poly_square(self):
        """
          Multiply poly by itself 
        """
        roots, inv_roots = ZField.get_roots()

        d1 = self.get_degree()
        dtp = 2*d1
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        # Recompute roots in case nroots changed or format.
        if len(roots) != dt or not isinstance(roots[0],type(self.zcoeff[0])):
            roots, inv_roots = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)
        self.expand_to_degree(dt, self)
        self._ntt(roots[:dt/2 + 1])
        for i in xrange(dt+1):
           self.zcoeff[i] = self.zcoeff[i] ** 2

        self._intt(inv_roots[:dt/2 + 1])
        self.expand_to_degree(dtp,self)

    def poly_mul_fft(self, p2, skip_fft=False): 
        """
          Multiply poly using FFT
        """

        roots, inv_roots = ZField.get_roots()

        d1 = self.get_degree()
        d2 = p2.get_degree()
        dtp = d1 + d2
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        # Recompute roots in case nroots changed or format.
        if len(roots) != dt or not isinstance(roots[0],type(self.zcoeff[0])):
            roots, inv_roots = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)

        self.expand_to_degree(dt, self)
        p2.expand_to_degree(dt, p2)

        if not skip_fft:
           self._ntt(roots[:dt/2 + 1])
           p2._ntt(roots[:dt/2 + 1])

        for i in xrange(dt+1):
           self.zcoeff[i] *= p2.zcoeff[i]

        self._intt(inv_roots[:dt/2 + 1])
        self.expand_to_degree(dtp,self)

    def ntt(self): 
        """
          FFT
        """
        roots,_ = ZField.get_roots()

        d1 = self.get_degree()
        dtp = d1
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        # Recompute roots in case nroots changed or format.
        if len(roots) != dt or not isinstance(roots[0],type(self.zcoeff[0])):
            roots, _ = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)

        self.expand_to_degree(dt, self)

        self._ntt(roots[:dt/2 + 1])

    def intt(self): 
        """
          IFFT
        """

        _,inv_roots = ZField.get_roots()

        d1 = self.get_degree()
        dtp = d1
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        # Recompute roots in case nroots changed or format.
        if len(inv_roots) != dt or not isinstance(inv_roots[0],type(self.zcoeff[0])):
            _, inv_roots = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)

        self.expand_to_degree(dt, self)
        self._intt(inv_roots[:dt/2 + 1])

    def _ntt(self, powtable):
        """
         Computes the forward number-theoretic transform of the given vector in place,
         with respect to the given primitive nth root of unity under the given modulus.
         The length of the vector must be a power of 2.

         Powtable is table with nth root roots of unity where n is the number of points in NTT
         Only N/2 roots of unity are needed

         NOTE https://www.nayuki.io/page/number-theoretic-transform-integer-dft
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
                    k += tablestep
            size *= 2

    def _intt(self, powtable):
        """
         Computes the inverse number-theoretic transform of the given vector in place,
         with respect to the given primitive nth root of unity under the given modulus.
         The length of the vector must be a power of 2.

         Powtable is table with nth root roots of unity where n is the number of points in NTT
         Only N/2 roots of unity are needed
        """
        self._ntt(powtable)
        nroots = ZFieldElExt(len(powtable)*2)
        if self.FIDX == ZUtils.FEXT:
           scaler = nroots.inv()
        else:
           scaler = nroots.inv().reduce()

        self.zcoeff = [c * scaler for c in self.get_coeff()]

    def poly_div(self, v):
        """
          Fast polynomial division ``u(x)`` / ``v(x)`` of polynomials with degrees
          m and n. Time complexity is ``O(n*log(n))`` if ``m`` is of the same order
          as ``n``.

           NOTE https://github.com/iden3/iden3js

           TODO optimize. There are far too many copies in this function and in multiplication
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
            t = ZPoly([ZPoly.one[self.FIDX]]).scale( 2 * ne) - s_copy

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
                me = rem.get_degree()
            else:
                done = True

        return q

    def inv(self):
        """
         Calculate the reciprocal of polynomial ``p(x)`` with degree ``k-1``,
         defined as: ``x^(2k-2) / p(x)``, where ``k`` is a power of 2.

         NOTE https://github.com/iden3/iden3js
         TODO invert is recursive -> change to iterative
        """
        k = self.get_degree() + 1
        assert k > 0 and self.zcoeff[-1] != 0 and 2 ** round(math.log(k, 2)) == k

        if k == 1:
            return ZPoly([self.zcoeff[0].inv()])

        npa = self.scale(-k/2)
        q = npa.inv()
        if self.FIDX == ZUtils.FEXT:
           a = (q << ZPoly.one[self.FIDX]).scale(3*k/2-2)
        else :
           a = ( q * ZPoly.two[self.FIDX]).scale(3*k/2-2)
        q.poly_mul(q)
        b = ZPoly(self)
        b.poly_mul(q)

        r = a - b 
        r.scale(-k + 2, r)

        return r

    def expand_to_degree(self, d, ret=None):
        """
         Extend list ``p`` representing a polynomial ``p(x)`` to
          match polynomials of degree ``d-1``. 

          If d > 0, fills higher degree coefficients with 0.
          If d < 0, removes higher order coefficients
        """
        if ret is not None:
            newP = ret
        else:
            newP = ZPoly(0)
        
        if d >= self.get_degree() :
            newP.zcoeff = self.get_coeff() + self.zero().get_coeff() * (d- self.get_degree())
        else:
            newP.zcoeff = self.get_coeff()[:d+1]

        newP.FIDX =   self.FIDX
        newP.degree = len(newP.zcoeff)-1

        return newP

    def scale(self, d, ret=None):
        """
          multiplies poly by x^d

          NOTE :  https://www.nayuki.io/page/number-theoretic-transform-integer-dft
        """
        if ret is not None:
            newP = ret
        else:
            newP = ZPoly(0)
        
        if d >= 0:
            newP.zcoeff = self.zero().get_coeff() * d + self.get_coeff()
        else:
            newP.zcoeff = self.get_coeff()[-d:]

        newP.FIDX =   self.FIDX
        newP.degree = len(newP.zcoeff)-1

        return newP

    def __rmul__(self, a):
        """
         Multiply scalar with polynomial 
        """
        return self * a

    def __mul__(self, a):
        """
         Multiply polynomial p(x) with scalar (constant) 
        """
        if isinstance(a, int) or isinstance(a, long) or isinstance(a, BigInt):
            return ZPoly([p * a for p in self.zcoeff])
        else:
            assert True, "Unexpected data type"

    def __add__(self, v):
        """
          Add polynomials ``u(x)`` and ``v(x)``.
        """
        if not isinstance(v, ZPoly):
            assert True, "Unexpected data type"
        elif isinstance(v,ZPolySparse):
            return v + self
        else :
            min_d = min(self.get_degree(), v.get_degree())
            if self.get_degree() >= v.get_degree():
                new_p = ZPoly(self.get_coeff())
            else:
                new_p = ZPoly(v)
    
            c1 = self.get_coeff()
            c2 = v.get_coeff()
    
            new_p.zcoeff[:min_d+1] = [c1[i] + c2[i] for i in xrange(min_d + 1)]
    
            return new_p.norm()

    def __sub__(self, v):
        """
          Sub polynomials ``u(x)`` and ``v(x)``.
        """
        return (self + (-v)).norm()

    def __neg__(self):
        """
          Negate polynomial coeffs
        """
        new_p = ZPoly(self)
        new_p_coeff = new_p.get_coeff()
        new_p.zcoeff = [-c for c in new_p_coeff]

        return new_p

    def __lshift__(self,k):
        """
          << K to all coeffs of poly
        """
        if isinstance(k,int) or isinstance(k,long) or isinstance(k, BigInt) and isinstance(self.zcoeff[0],ZFieldElExt):
            return ZPoly([c << k for c in self.get_coeff()])
        else:
            assert True, "Unexpected data type"

    def zero(self):
        """
          return 0 poly
        """
        if self.FIDX == ZUtils.FEXT:
            return ZPoly([ZFieldElExt(0)])
        else:
            return ZPoly([ZFieldElRedc(0)])


    def show(self):
        """
         Print poly
        """
        if type(self.get_coeff()) is list:
          print [c.as_long() for c in self.get_coeff()]
        else :
          l =  self.dict_to_list()
          print [c.as_long() for c in l]


class ZPolySparse(ZPoly):
    def __init__(self, p):
        super(ZPolySparse, self).__init__(p)

    def to_sparse(self):
        """
          Convert to sparse. Do nothing
        """
        return self

    def to_dense(self):
        """
          Convert to dense poly representation
        """
        zcoeffs = self.dict_to_list()

        return ZPoly(zcoeffs)


    def norm(self):
        """
        Normalize poly to have a non zero most significant coefficient
        """
        newP = ZPolySparse(self)
        coeff = sorted([long(k) for k in newP.zcoeff.keys()])

        if coeff[-1] != 0:
                return newP

        for c in coeff[::-1]:
            if newP.zcoeff[str(c)] == 0:
                del newP.zcoeff[str(c)]
            else:
                break

        newP.degree = len(newP.zcoeff.keys())-1
        if newP.degree > 0:
            return newP
        else :
            return self.zero()
        

    def poly_mul(self, p2):
        """
          Not supported
        """
        assert True, "Operation not supported"
        
    def poly_mul_normal(self, p2):
        """
          Not supported
        """
        assert True, "Operation not supported"

    def poly_square(self):
        """
          Not supported
        """
        assert True, "Operation not supported"

    def poly_mul_fft(self, p2): 
        """
          Not supported
        """
        assert True, "Operation not supported"

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
            return ZPolySparse([self.zcoeff[k] * a for k in self.zcoeff.keys()])
        else:
            assert True, "Unexpected data type"

    def ntt(self, powtable):
        """
          Not supported
        """
        assert True, "Operation not supported"

    def intt(self, powtable):
        """
          Not supported
        """
        assert True, "Operation not supported"

    def expand_to_degree(self, d, ret=None):
        """
          Not supported
        """
        assert True, "Operation not supported"

    def scale(self, d, ret=None):
        """
          Not supported
        """
        assert True, "Operation not supported"

    def __add__(self, v):
        """
          Add polynomials ``u(x)`` and ``v(x)``.
        """
        if not isinstance(v, ZPoly):
            assert True, "Unexpected data type"

        elif isinstance(v,ZPolySparse):   
            # sparse + sparse -> sparse
            newP = ZPolySparse(self)

            for k in v.zcoeff.keys():
                if k in newP.zcoeff:
                   newP.zcoeff[k] += v.zcoeff[k]
                else:
                   newP.zcoeff[k] = v.zcoeff[k]
        else :
            # sparse + dense -> dense
            newP = ZPoly(v)
            coeff = sorted([long(k) for k in p.keys()])
            if coeff[-1] > newP.get_degree():
                newP.expand_to_degree(c[-1])

            for c in coeff:
                newP[c] += v[str(c)]
        newP.norm()

        return new_p

    def __neg__(self):
        """
          Negate polynomial coeffs
        """
        return ZPolySparse({i: -p[i] for i in p.keys()})

    def __lshift__(self,k):
        """
        """
        if isinstance(k,int) or isinstance(k,long) or isinstance(k, BigInt) and isinstance(self.zcoeff[0],ZFieldElExt):
            return ZPolySparse([self.zcoeff[k] << k for k in self.zcoeff.keys()])
        else:
            assert True, "Unexpected data type"

    def inv(self):
        """
          Not supported
        """
        assert True, "Operation not supported"

    def poly_div(self, v):
        """
          Not supported
        """
        assert True, "Operation not supported"

