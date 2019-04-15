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
import numpy as np
import os,sys, os.path

from zfield import *

class ZPoly(object):
    # beyond this degree, poly mul is done with FFT
    FFT_MUL_THRES = 128

    one = [None, None]
    two = [None, None]

    init_zpoly = False


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
        if not ZField.is_init():
            assert False, "Prime field not initialized"
        elif isinstance(p, int) or isinstance(p, long) or isinstance(p, BigInt):
            if p < 0:
                assert False, "Polynomial needs to be at least degree 0"
            else:
                self.degree = p;
                prime = ZField.get_extended_p()
                self.zcoeff = [ZFieldElExt(randint(0, prime.as_long() - 1)) for x in xrange(p + 1)]
                self.zcoeff[-1] = ZFieldElExt(randint(1, prime.as_long() - 1))
                self.FIDX = ZUtils.FEXT
        elif type(p) is list and isinstance(self, ZPolySparse) :
            assert False, "Unexpected data type"
        #elif type(p) is list or type(p) is dict:
        #elif type(p) is list or isinstance(self, ZPolySparse):
        elif type(p) is list or type(p) is dict:
            self.zcoeff, self.degree, self.FIDX = ZPoly.set_properties(p)
        elif isinstance(p, ZPoly):
            self.degree = p.get_degree()
            self.zcoeff = p.get_coeff()
            self.FIDX = p.FIDX
        else:
            assert False, "Unexpected data type"

        if ZPoly.init_zpoly == False or force_init:
            ZPoly.init_constants()

    @classmethod
    def init_constants(cls):
        ZPoly.one = [ZFieldElExt(1), ZFieldElExt(1).reduce()]
        ZPoly.two = [ZFieldElExt(2), ZFieldElExt(2).reduce()]
        ZPoly.init_zpoly = True

    @classmethod
    def init(cls, idx=None):
        old_idx = ZField.active_prime_idx
        if idx is not None:
            ZField.set_field(idx)
        ZPoly.init_constants()
        ZField.active_prime_idx = old_idx

    @classmethod
    def set_properties(cls, p):
        """
         Retrieve poly properties coeff, degree and FIDX from list/dictionaty with coefficients.

         p : List/Dict with coefficients

        return zcoeff, degree, FIDX
        """
        if type(p) is list:
            if len(p) == 0:
              degree = 0
              zcoeff = [0]
              FIDX = ZUtils.FEXT
              return zcoeff, degree, FIDX
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
                assert False, "Unexpected data type"
        elif type(p) is dict:
            c = sorted([long(k) for k in p.keys()])
            if len(c) > 0:
                degree = long(c[-1])
            else:
                # TODO : Improve this. IT will fail if coeffs are ZFieldElRedc. Also, I am using
                # this case to solve issue win groth protocol when array y sparse polys includes
                # empty case. I should solve that in a different way
                p = {'0':0 }
                degree = 0
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
                assert False, "Unexpected data type"

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
        assert False, "Operation not supported"

    def as_list(self):
        if type(self.zcoeff) == list:
            return [x.as_long() for x in self.zcoeff]

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
            assert False, "Unexpected data type"
        
        if self.FIDX != p2.FIDX:
            assert False, "Coefficients need to be in the same format"
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

        elif d1+d2 < ZPoly.FFT_MUL_THRES:
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
         p1 = np.polymul(p1[::-1], p2[::-1])

         if self.FIDX == ZUtils.FEXT:
            self.zcoeff = [ZFieldElExt(c) for c in p1[::-1]]
         else:
            self.zcoeff = [ZFieldElExt(c).reduce() for c in p1[::-1]]

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

        inv_roots_slice = inv_roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]
        roots_slice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]

        # Recompute roots in case nroots changed or format.
        #TODO
        #if len(roots) != dt or not isinstance(roots[0],type(self.zcoeff[0])):
            #roots, inv_roots = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)
        self.expand_to_degree(dt, self)
        self._ntt(roots_slice[:dt/2 + 1])
        #self.zcoeff = np.multiply(self.zcoeff, self.zcoeff).tolist()
        for i in xrange(dt+1):
           self.zcoeff[i] = self.zcoeff[i] ** 2

        self._intt(inv_roots_slice[:dt/2 + 1])
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

        inv_roots_slice = inv_roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]
        roots_slice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]
        # Recompute roots in case nroots changed or format.
        #TODO
        #if len(roots) != dt+1 or not isinstance(roots[0],type(self.zcoeff[0])):
        #    roots, inv_roots = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)

        self.expand_to_degree(dt, self)
        p2.expand_to_degree(dt, p2)

        if not skip_fft:
           self._ntt(roots_slice[:dt/2 + 1])
           p2._ntt(roots_slice[:dt/2 + 1])

        self.zcoeff = np.multiply(self.zcoeff, p2.zcoeff).tolist()
        #for i in xrange(dt+1):
        #   self.zcoeff[i] *= p2.zcoeff[i]

        self._intt(inv_roots_slice[:dt/2 + 1])
        self.expand_to_degree(dtp,self)

    @classmethod
    def printM(cls,M):
        for i in xrange(M.shape[0]):
            for j in xrange(M.shape[1]):
                print M[i,j].as_long(),
            print "\n"

    def ntt_parallel2D(self,nrows,ncols):
        """
           Parallel N point FFT
             1) Decompose N points into a n1xn2 matrix, filling it in column order
             2) Perform n1 FFT of n2 points each and put results in n1xn2 matrix
             3) Multiply resulting matrix elements Ajk by root+-j*k (+ FFT/- IFFT)
             4) Transpose matrix to n2xn1
             5) Perform n2 FFT of n1 points
        """
        roots, _ = ZField.get_roots()

        dt = nrows*ncols-1
        self.expand_to_degree(dt, self)

        M = np.asarray(self.zcoeff).reshape((nrows, ncols), order='F')

        roots_Mslice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]

        roots_Nslice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(ncols)]

        #print M.shape
        #print ZPoly(roots_Nslice).as_uint256()
        for i,rows in enumerate(M):
            newP = ZPoly(rows.tolist())
            #newP._ntt_DIF(roots_Nslice[:ncols/2+1])
            #print "IN ROW" +str(i) +": " + str(newP.as_uint256())
            newP._ntt(roots_Nslice[:ncols/2+1])
            #print "OUT ROW" +str(i) +": " + str(newP.as_uint256())
            for k,c in enumerate(newP.get_coeff()):
                M[i,k] = c * roots_Mslice[i*k]
        
        M = M.transpose()

        roots_Nslice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(nrows)]

        for i,rows in enumerate(M):
            newP = ZPoly(rows.tolist())
            #print "IN ROW" +str(i) +": " + str(newP.as_uint256())
            #newP._ntt_DIF(roots_Nslice[:nrows/2+1])
            newP._ntt(roots_Nslice[:nrows/2+1])
            #print "OUT ROW" +str(i) +": " + str(newP.as_uint256())
            M[i] = newP.get_coeff()

        self.zcoeff = np.reshape(M,-1,order='F').tolist()

    def intt_parallel2D(self,nrows,ncols):
        """
           Parallel N point FFT
             1) Decompose N points into a n1xn2 matrix, filling it in column order
             2) Perform n1 FFT of n2 points each and put results in n1xn2 matrix
             3) Multiply resulting matrix elements Ajk by root+-j*k (+ FFT/- IFFT)
             4) Transpose matrix to n2xn1
             5) Perform n2 FFT of n1 points
        """
        _,  inv_roots = ZField.get_roots()

        dt = nrows*ncols-1
        self.expand_to_degree(dt, self)

        M = np.asarray(self.zcoeff).reshape((nrows, ncols), order='F')

        inv_roots_Mslice = inv_roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]

        inv_roots_Nslice = inv_roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(ncols)]
        scaler = ZFieldElExt(len(inv_roots_Mslice)).inv().reduce()

        for i,rows in enumerate(M):
            newP = ZPoly(rows.tolist())
            #newP._ntt_DIF(inv_roots_Nslice[:ncols/2+1])
            newP._ntt(inv_roots_Nslice[:ncols/2+1])
            for k,c in enumerate(newP.get_coeff()):
                M[i,k] = c * inv_roots_Mslice[i*k]

        M = M.transpose()

        inv_roots_Nslice = inv_roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(nrows)]

        for i,rows in enumerate(M):
            newP = ZPoly(rows.tolist())
            #print "IN ROW" +str(i) +": " + str(newP.as_uint256())
            #newP._ntt_DIF(inv_roots_Nslice[:nrows/2+1])
            newP._ntt(inv_roots_Nslice[:nrows/2+1])
            #print "OUT ROW" +str(i) +": " + str(newP.as_uint256())
            for k,c in enumerate(newP.get_coeff()):
                M[i,k] = c * scaler
            #M[i] = newP.get_coeff()

        self.zcoeff = np.reshape(M,-1,order='F').tolist()


    
    def ntt_parallel3D(self,nrows,depthr,ncols,depthc):
        """
           Parallel N point FFT
             1) Decompose N points into a n1xn2 matrix, filling it in column order
             2) Perform n1 FFT of n2 points each and put results in n1xn2 matrix
             3) Multiply resulting matrix elements Ajk by root+-j*k (+ FFT/- IFFT)
             4) Transpose matrix to n2xn1
             5) Perform n2 FFT of n1 points
        """
        roots, _ = ZField.get_roots()

        dt = nrows*ncols*depthc*depthr-1
        self.expand_to_degree(dt, self)

        M = np.asarray(self.zcoeff).reshape((nrows*depthr, ncols*depthc), order='F')

        roots_Mslice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]

        if depthc != 1:
            for i,rows in enumerate(M):
                newP = ZPoly(rows.tolist())
                newP.ntt_parallel2D(ncols,depthc)
    
                for k,c in enumerate(newP.get_coeff()):
                    M[i,k] = c * roots_Mslice[i*k]
        else:
            roots_Nslice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(ncols)]
            for i,rows in enumerate(M):
                newP = ZPoly(rows.tolist())
                newP._ntt_DIF(roots_Nslice[:ncols/2+1])
    
                for k,c in enumerate(newP.get_coeff()):
                    M[i,k] = c * roots_Mslice[i*k]

        M = M.transpose()

        if depthr != 1:
            for i,rows in enumerate(M):
                newP = ZPoly(rows.tolist())
                newP.ntt_parallel2D(nrows,depthr)
                M[i] = newP.get_coeff()
        else:
            roots_Nslice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(nrows)]
            for i,rows in enumerate(M):
                newP = ZPoly(rows.tolist())
                newP._ntt_DIF(roots_Nslice[:nrows/2+1])
                M[i] = newP.get_coeff()

        self.zcoeff = np.reshape(M,-1,order='F').tolist()

    def ntt(self): 
        """
          FFT
        """
        roots,_ = ZField.get_roots()

        d1 = self.get_degree()
        dtp = d1
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        roots_slice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]

        # Recompute roots in case nroots changed or format.
        #if len(roots) != dt+1 or not isinstance(roots[0],type(self.zcoeff[0])):
        #    roots, _ = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)

        self.expand_to_degree(dt, self)

        self._ntt(roots_slice[:dt/2 + 1])

    def ntt_DIF(self): 
        """
          FFT
        """
        roots,_ = ZField.get_roots()

        d1 = self.get_degree()
        dtp = d1
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        roots_slice = roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]

        # Recompute roots in case nroots changed or format.
        #if len(roots) != dt+1 or not isinstance(roots[0],type(self.zcoeff[0])):
        #    roots, _ = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)

        self.expand_to_degree(dt, self)

        self._ntt_DIF(roots_slice[:dt/2 + 1])

  
    def intt(self): 
        """
          IFFT
        """

        _,inv_roots = ZField.get_roots()

        d1 = self.get_degree()
        dtp = d1
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        inv_roots_slice = inv_roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]

        # Recompute roots in case nroots changed or format.
        #TODO
        #if len(inv_roots) != dt+1 or not isinstance(inv_roots[0],type(self.zcoeff[0])):
        #    _, inv_roots = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)

        self.expand_to_degree(dt, self)
        self._intt(inv_roots_slice[:dt/2 + 1])

    def intt_DIT(self): 
        """
          IFFT
        """

        _,inv_roots = ZField.get_roots()

        d1 = self.get_degree()
        dtp = d1
        dt = (1 << long(math.ceil(math.log(dtp+1, 2)))) - 1

        inv_roots_slice = inv_roots[0:ZUtils.NROOTS:ZUtils.NROOTS/(dt+1)]

        # Recompute roots in case nroots changed or format.
        #TODO
        #if len(inv_roots) != dt+1 or not isinstance(inv_roots[0],type(self.zcoeff[0])):
        #    _, inv_roots = ZField.find_roots(dt+1, rformat_ext = self.FIDX==ZUtils.FEXT)

        self.expand_to_degree(dt, self)
        self._intt_DIT(inv_roots_slice[:dt/2 + 1])



    def _ntt_DIF(self, powtable):
        """
         Computes the forward number-theoretic transform of the given vector in place ,
         with respect to the given primitive nth root of unity under the given modulus.
         The length of the vector must be a power of 2.
         INput is ordered. Output is unordered

         Powtable is table with nth root roots of unity where n is the number of points in NTT
         Only N/2 roots of unity are needed
        """
        vector = self.zcoeff
        n = len(vector).bit_length()-1
   
        for i in range(n):
             for j in range(2**i):
                 for k in range(2**(n-i-1)):
                     s = j*2**(n-i) + k
                     t = s + 2 ** (n-i-1)
                     left = vector[s]
                     right = vector[t] 
                     vector[s] = left + right
                     vector[t] = (left - right) * powtable[(2**i)*k]
                     #print "L:" + str(i)
                     #print "s:" + str(s) + " " + str(left.as_uint256()) +"   -> " +str(vector[s].as_uint256())
                     #print "t:" + str(t) + " " + str(right.as_uint256()) +"  -> " +str(vector[t].as_uint256())
                     #print "k:" + str((2**i)*k) + " root " +str(powtable[(2**i)*k].as_uint256())

    def _intt_DIT(self, powtable):
        """
         Computes the inverse number-theoretic transform of the given vector in place,
         with respect to the given primitive nth root of unity under the given modulus.
         The length of the vector must be a power of 2.

         Powtable is table with nth root roots of unity where n is the number of points in NTT
         Only N/2 roots of unity are needed
        """
        vector = self.zcoeff
        n = len(vector).bit_length()-1
   
        for i in range(n-1,-1,-1):
             for j in range(2**i):
                 for k in range(2**(n-i-1)):
                     s = j*2**(n-i) + k
                     t = s + 2 ** (n-i-1)
                     left = vector[s]
                     right = vector[t] 
                     vector[s] = left + right * powtable[(2**i)*k]
                     vector[t] = left - right * powtable[(2**i)*k]
                     #print "L:" + str(i)
                     #print "s:" + str(s) + " " + str(left.as_uint256()) +"   -> " +str(vector[s].as_uint256())
                     #print "t:" + str(t) + " " + str(right.as_uint256()) +"  -> " +str(vector[t].as_uint256())
                     #print "k:" + str((2**i)*k) + " root " +str(powtable[(2**i)*k].as_uint256())

        nroots = ZFieldElExt(len(powtable)*2)
        if self.FIDX == ZUtils.FEXT:
           scaler = nroots.inv()
        else:
           scaler = nroots.inv().reduce()

        self.zcoeff = [c * scaler for c in self.get_coeff()]
 
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

    def poly_div(self, v, invpol=None):
        """
          Fast polynomial division ``u(x)`` / ``v(x)`` of polynomials with degrees
          m and n. Time complexity is ``O(n*log(n))`` if ``m`` is of the same order
          as ``n``.

           NOTE https://github.com/iden3/iden3js

           TODO optimize. There are far too many copies in this function and in multiplication
        """
        if not isinstance(v, ZPoly):
            assert False, "Unexpected data type"

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

        if invpol is None:
           s = ve.inv()
        else:
           s = invpol

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

    def poly_div_snarks(self, n):
        """
          Fast polynomial division ``u(x)`` / ``v(x)`` of polynomials with degrees
          m and n. Time complexity is ``O(n*log(n))`` if ``m`` is of the same order
          as ``n``. Assumes v(x) is of the form x^n - 1

        """
        m = self.get_degree()

        if m < n:
            return self.zero()

        # ensure deg(v) is one less than some power of 2
        # by extending v -> ve, u -> ue (mult by x^nd)
        nd = (1<<  int(math.ceil(math.log(n+1, 2))) )- 1 - n
        ue = self.scale(nd)
        me = m + nd
        ne = n + nd

        # handle the case when m>2n
        q = self.zero()
        rem = ZPoly(ue)
        done = False
        niter = 0

        while not done:
            if len(rem.get_coeff()[2*ne-nd:]) == 0:
                return q
            us = ZPoly(rem.get_coeff()[ne:]) + ZPoly(rem.get_coeff()[2*ne-nd:]) # degree me - ne
            q = q + us

            if me > 2 * ne:
                rem = ZPoly(rem.get_coeff()[ne+1:])
                me = rem.get_degree()
            else:
                done = True
            niter +=1
            #if niter==2:
            #  return q

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
            if a == 0 and self.FIDX == ZUtils.FEXT:
               return ZPoly([0])
            elif a == 0 and self.FIDX == ZUtils.FRDC:
               return ZPoly([ZFieldElRedc(0)])
            elif a == 1:
               return ZPoly(self)
            else:
               return ZPoly([p * a for p in self.zcoeff])
        else:
            assert False, "Unexpected data type"

    def __add__(self, v):
        """
          Add polynomials ``u(x)`` and ``v(x)``.
        """
        if not isinstance(v, ZPoly):
            assert False, "Unexpected data type"
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
            assert False, "Unexpected data type"

    def __eq__(self, v):
        if not isinstance(v, ZPoly):
            assert False, "Unexpected data type"
        p1 = self.norm()
        p2 = v.norm()
        if p1.degree != p2.degree :
            return False

        for idx in xrange(p1.degree+1):
           if p1.zcoeff[idx] != p2.zcoeff[idx]:
               return False

        return True

    def __ne__ (self,v):
        return not self == v

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

    def as_uint256(self):
        if type(self.get_coeff()) is list:
            return np.asarray([c.as_uint256() for c in self.get_coeff()])
        else :
            Val = []
            Coeff = []
            for k,v in self.zcoeff.items():
              Coeff.append(np.uint32(k ))
              Val.append(v.as_uint256())
            Val = np.concatenate(Val)
            return len(self.zcoeff), np.concatenate((np.asarray(Coeff,dtype=np.uint32), Val))
            #return np.asarray([np.concatenate([(np.uint32(k )], v.as_uint256())) for k,v in self.zcoeff.items()], dtype=np.uint32)

    @staticmethod
    def from_uint256(x,reduced=False):
        if reduced:
            P = ZPoly([ZFieldElRedc(BigInt.from_uint256(x_)) for x_ in x])
        else:
            P = ZPoly([ZFieldElExt(BigInt.from_uint256(x_)) for x_ in x])

        return P


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
        
    def extend(self):
        """
          Convert poly coeffs from REDC to EXT 
        """
        if self.FIDX == ZUtils.FRDC:
            return ZPolySparse({k : v.extend() for k,v in self.zcoeff.items()})
        else:
            return self

    def reduce(self):
        """
          Convert poly coeffs from EXT to REDC
        """
        if self.FIDX == ZUtils.FEXT:
            return ZPolySparse({k : v.reduce() for k,v in self.zcoeff.items()})
        else:
            return self

    def poly_mul(self, p2):
        """
          Not supported
        """
        assert False, "Operation not supported"
        
    def poly_mul_normal(self, p2):
        """
          Not supported
        """
        assert False, "Operation not supported"

    def poly_square(self):
        """
          Not supported
        """
        assert False, "Operation not supported"

    def poly_mul_fft(self, p2):
        """
          Not supported
        """
        assert False, "Operation not supported"

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
            if a == 0 and self.FIDX == ZUtils.FEXT:
               return ZPolySparse({'0':0})
            elif a == 0 and self.FIDX == ZUtils.FRDC:
               return ZPolySparse({'0':ZFieldElRedc(0)})
            elif a == 1:
               return ZPolySparse(self)
            else:
               return ZPolySparse({k : self.zcoeff[k] * a for k in self.zcoeff.keys()})
        else:
            assert False, "Unexpected data type"

    def ntt(self, powtable):
        """
          Not supported
        """
        assert False, "Operation not supported"

    def intt(self, powtable):
        """
          Not supported
        """
        assert False, "Operation not supported"

    def expand_to_degree(self, d, ret=None):
        """
          Not supported
        """
        assert False, "Operation not supported"

    def scale(self, d, ret=None):
        """
          Not supported
        """
        assert False, "Operation not supported"

    def __add__(self, v):
        """
          Add polynomials ``u(x)`` and ``v(x)``.
        """
        if not isinstance(v, ZPoly):
            assert False, "Unexpected data type"

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
            self_coeff = self.get_coeff()
            self_coeff_deg = sorted([long(k) for k in self_coeff.keys()])
            if self_coeff_deg[-1] > newP.get_degree():
                newP = newP.expand_to_degree(self_coeff_deg[-1])

            for c in self_coeff_deg:
                newP.zcoeff[c] += self_coeff[str(c)]

        newP.degree = max(self.get_degree(),v.get_degree())
        return newP.norm()

    def __eq__(self,other):
        return self.FIDX == other.FIDX and self.zcoeff == other.zcoeff and self.degree == other.degree

    def __ne__(self,other):
        return not self==other

    def __neg__(self):
        """
          Negate polynomial coeffs
        """
        return ZPolySparse({i: -self.zcoeff[i] for i in self.zcoeff.keys()})

    def __lshift__(self,x):
        """
        """
        if isinstance(x,int) or isinstance(x,long) or isinstance(x, BigInt) and isinstance(self.zcoeff[0],ZFieldElExt):
            return ZPolySparse({k : self.zcoeff[k] << x for k in self.zcoeff.keys()})
        else:
            assert False, "Unexpected data type"

    def inv(self):
        """
          Not supported
        """
        assert False, "Operation not supported"

    def poly_div(self, v):
        """
          Not supported
        """
        assert False, "Operation not supported"


