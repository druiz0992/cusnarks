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
# https://www.nayuki.io/page/montgomery-reduction-algorithm
# https://www.nayuki.io/page/number-theoretic-transform-integer-dft

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : zfield
//
// Date       : 14/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Implementation of finite field functionality
//   
//   1) ZField must be initialized (only with a prime makes sense, although it is not 
//           verified). It doesn't return anything.
//       Methods:
//          @classmethod get(cls) : Returns prime
//          @classmethod set(cls,p) : Changes prime - Not recommended to change prime. ZFieldEl
//             initialied with previous prime will stop making sense.
//          @classmethod is_init(cls) : True if ZField is initialized
//          @classmethod is_redc(cls) : True if ZField prime has been reduced (Montgomery reduction)
//          @classmethod show(cls) : Print ZField prime
//          @classmethod reduce(cls) : Reduce ZField prime (Montgomery)
//          @classmethod extend(cls) : Extend ZField prime
//          @classmethod find_roots(cls,nroots, find_inv_roots=False) : Returns primitive nth root 
//          
//   2) ZFieldEl implements finite field arithmetic. Assumes that ZField has been initialized. Extends BigInt class
//
//   TODO
//    - ZField only makes sense if initialized with prime.. Not checked
//    
// ------------------------------------------------------------------

"""
from bigint import BigInt
import math
import numpy as np
from random import randint

class ZField:
  PRIME_THR = long(1e10)
  BN128_DATA = {'curve' : "BN128",
                'prime' : 21888242871839275222246405745257275088548364400416034343698204186575808495617L,
                'factor_data' : { 'factors' : [2,3,13,29,983,11003,237073, 405928799L, 1670836401704629L, 13818364434197438864469338081L],
                                        'exponents' : [28,2,1,1,1,1,1,1,1,1] }
               }

  P1009_DATA = {'curve' : "P1009",
                'prime' : 1009,
                'factor_data' : { 'factors' : [2,3,7],
                                  'exponents' : [4,2,1] }
               }
  init_prime = False    # Flag : Is prime initialized
  ext_prime  = None     # Field prime
  redc_prime = None     # Field Prime (montgomery reduced)
  roots = []            # Field roots of unity
  inv_roots = []        # Filed inverse roots of unity

  # Montgomery reduction data
  redc_data = {'bitlen' : 0, 'R' : 0, 'Rmask' : 0, 'Rp' : 0, 'Pp' : 0, 'convertedone' : 0}

  # Factorization data
  factor_data = {'factors' :   [],  # prime factors of prime - 1
                 'exponents' : []   # prime factors exponents of prime  - 1
               }

  def __init__(self,q,factor_data=None):
      """
      Constructor

      Parameters:
      ----------
        q :           BigInt, long, int, string integer that initializes 
                          Field characteristic
        factor_data : (Optional) String or Dictionary with factorization information 
                         for the provided field characteristic. If characteristic
                         is smaller than PRIME_THR, data can be computed. Else
                         it needs to be provided.
                         if String format, expected string is the name of the predefined data
                            P1009 or BN128
                         Contents of dictionary:
           'factors'  : list of length N of int/long containing prime factors of prime-1
           'exponents': list of length N of int/long containing exponents of prime factors of 
                            prime-1 such that 
                            prime - 1 = (factor[0] ** exponent[0]) * 
                                             (factor[1] ** exponent[1]) * 
                                             (factor[N-1] ** exponent[N-1]) 
      """
      ZField.init_prime = True
      ZField.ext_prime = BigInt(q)
      # Apply processing for future reduction
      ZField.reduce()
      # init factorization data
      ZField.factorize(factor_data)

  @classmethod
  def get_extended_p(cls):
    """
      Returns extended prime as BigNum
    """
    return ZField.ext_prime

  @classmethod
  def get_reduced_p(cls):
    """
      Returns reduced prime as BigNum. 
       If None, it means that Field has't undergone Montgomery reduction 
    """
    return ZField.redc_prime

  @classmethod
  def get_reduce_data(cls):
    """
      Returns reduce data
    """
    return ZField.redc_data

  @classmethod
  def is_init(cls):
      """
        True if ZField has been intialized.
         False otherwise
      """
      return ZField.init_prime

  @classmethod
  def reduce(cls):
      """
        Montgomery reduction. Find alternative version of modulus R to speed up multiplications instead of using P (prime)
          if R * Rp = 1 (mod P) =>  R * Rp - P * Pp = 1.  R is a power of 2 larger than P
            Rp and Pp are found using extended euclidian algo.

          Assume that we want to multiple a * b = c (mod P)
            _a = a * R (mod P)
            _b = b * R (mod P)  
               then, Montgomery product computes _c = a_ * _b * Rp (mod P) = a * b * R (mod P)  only dividing by R which is fast
            c = a * b (mod P) = _c * Rp (mod P)

      """
      p = ZField.get_extended_p().as_long()
      bitlen = int(math.ceil(math.log(p,2)))
      t = 1 << bitlen | 1 # force it to be odd
      ZField.redc_data['bitlen'] = (t.bit_length() // 8 +1 )* 8 # Multiple of 8
      ZField.redc_data['R'] = long(1 << ZField.redc_data['bitlen'])
      ZField.redc_data['Rmask'] = long(ZField.redc_data['R'] - 1)
      ZField.redc_data['Rp'] = ZField.inv(ZField.redc_data['R'] % p).as_long()
      ZField.redc_data['Pp'] = (ZField.redc_data['R'] * ZField.redc_data['Rp'] - 1) // p
      ZField.redc_data['convertedone'] = ZField.redc_data['R'] % p
      ZField.redc_prime = BigInt(ZField.redc_data['R'])

  @staticmethod
  def inv(x):
     """
       returns X' such that X * X' = 1 (mod p)
     """
     if not ZField.is_init():
         assert True, "Finite field not initialized"
     elif isinstance(x,BigInt) or isinstance(x,ZFieldEl):
         x_l = x.bignum
     elif isinstance(x,int)  or isinstance(x,long)   :
         x_l = x
     else :
         assert True, "Invalid type"

     p = ZField.get_extended_p().as_long()
     # Based on a simplification of the extended Euclidean algorithm
     assert p > 0 and 0 <= x_l < p 
     y = x_l
     x_l = p
     a = 0
     b = 1
     while y != 0:
        a, b = b, a - x_l // y * b
        x_l, y = y, x_l % y
     if x_l == 1:
         return ZFieldEl(a % p)
     else:
        raise ValueError("Reciprocal does not exist")

  @classmethod
  def find_generator(cls):
      """
      Returns an arbitrary generator gen of the multiplicative group ZField with characteristic p
        where gen  ^ (k*nroots) = 1 mod p. If p is prime, an answer must exist
      """
      if not ZField.is_init():
         assert True, "Finite field not initialized"

      gamma=long(1)
      prime = ZField.get_extended_p().as_long()
      for i in xrange(len(ZField.factor_data['factors'])):
           beta = long(1)
           prime_factor = ZField.factor_data['factors'][i]
           exponent = ZField.factor_data['exponents'][i]
              
           # alpha is random number between 0 and mod (inclusive)
           while beta == 1:
             alpha = randint(0,prime)
             beta = pow(alpha,(prime-1)/prime_factor,prime)

           gamma = gamma * pow(alpha,(prime-1)/(prime_factor**exponent),prime)
           gamma = gamma % (prime)
       
      return ZFieldEl(gamma)

  @classmethod
  def find_primitive_root(cls, nroots):
      """
        Returns primitive root such that root = gen ^ nroots % prime,
      """
      if not ZField.is_init():
         assert True, "Finite field not initialized"

      gen = ZField.find_generator().as_long()
      prime = ZField.get_extended_p().as_long()
      return ZFieldEl(pow(gen, (prime - 1) // nroots, prime))

  @classmethod
  def get_roots(cls):
      """
        returns computed roots of unity
      """
      return ZField.roots, ZField.inv_roots

  @classmethod
  def find_roots(cls, nroots, find_inv_roots=True):
      """
        Computes and returns nroots of unity. If find_inv_roots is True, inverse roots are
          also computed
      """
      if not ZField.is_init():
        assert True, "Prime not initialized"

      # initialize roots
      ZField.roots = [ ZFieldEl(1), ZField.find_primitive_root(nroots) ]

      ZField.inv_roots = []
      root_1 = ZField.roots[1]
      for i in xrange(nroots-2):
         ZField.roots.append(root_1 * ZField.roots[-1])

      if find_inv_roots:
         ZField.inv_roots = [ZFieldEl(1)]
         ZField.inv_roots[1:] =  map(ZField.inv, ZField.roots[1:] )

      return ZField.roots, ZField.inv_roots

  @classmethod
  def factorize(cls, factor_data=None):
      """
       Factorizes prime - 1. Only works for small primes less than PRIME_THR.
      """
      if not ZField.is_init():
         assert True, "Finite field not initialized"
      elif isinstance(factor_data,dict) and 'factors' in factor_data and 'exponents' in factor_data:
           ZField.factor_data = factor_data
      
      elif isinstance(factor_data,str):
        if factor_data == ZField.BN128_DATA['curve']:
           ZField.factor_data = ZField.BN128_DATA['factor_data']
        elif factor_data == ZField.P1009_DATA['curve']:
           ZField.factor_data = ZField.P1009_DATA['factor_data']
        else:
          assert True, "Curve information not available"
          
      elif ZField.get_extended_p() > ZField.PRIME_THR :
        assert True,"Prime is too large to factorize"

      else:
        prime_1 = ZField.get_extended_p().as_long() - 1
        ZField.factor_data = ZField.prime_factors(prime_1)
 
      return ZField.factor_data
     
  @classmethod
  def get_factors(cls):
      return ZField.factor_data

  @classmethod
  def prime_factors(cls,n):
     """
      Factorizes number into prime factors. Returns dictionary with following info:
       'factors' : array of prime factors
       'exponents' : array of prime factor exponent
     """
     if n < 1:
         assert True,  "Number needs to be larger than 1"
     result = {'factors' :[], 'exponents' : [] }
     i = 2
     number = n
     end = math.sqrt(n)
     while i <= end:
         if n % i == 0:
             n //= i
             result['factors'].append(i)
             result['exponents'].append(1)
             while n % i == 0:
                 n //= i
                 result['exponents'][-1]+=1
             end = math.sqrt(n)
         i += 1
     if n > 1:
         result['factors'].append(n)
         result['exponents'].append(1)
         while n % i == 0 and n > 0:
             n //= i
             result['exponents'][-1]+=1
     assert np.prod([r**e for r,e in zip(result['factors'],result['exponents'])]) == number

     return result

class ZFieldEl(BigInt):
   def __init__(self, bignum):
     """
      Initialization

      Parameters
      -----------
       bignum 
     """
     if ZField.is_init():
         BigInt.__init__(self,bignum % ZField.get_extended_p().as_long())
         #if ZField.is_redc():
         #  self.reduce()
         #  self.is_redc = True
         #else:
         #    self.is_redc = False
     else :
       assert True, "Prime not initialized"

   def reduce(self):
       if not self.is_redc:
           self.bignum = (self.bignum << ZField.redc_data['bitlen']) % ZField.get_extended_p()
           self.is_redc = True

   def extend(self):
       if self.is_redc:
           self.bignum = (self.bignum * ZField.redc_data['Rp']) % ZField.get()
           self.is_redc = False

   # Arithmetic operators
   # +, -, *, /, %, pow
   def __add__(self,x):
    """
     TODO : numbers need to be in the same format (reduced/extended). Program doesn't
       check for this condition
    """
    if isinstance(x,BigInt):
      return (self.bignum + x.bignum) % ZField.get()
    elif isinstance(x,int) or isinstance(x,long):
      return (self.bignum + x) % ZField.get()
    else :
      assert True,"Invalid type"

   def __sub__(self,x):
    """
     TODO : numbers need to be in the same format (reduced/extended). Program doesn't
       check for this condition
    """
    if isinstance(x,BigInt):
     return (self.bignum - x.bignum) % ZField.get()
    elif isinstance(x,int) or isinstance(x,long):
      return (self.bignum - x) % ZField.get()

   def __mul__(self,x):
    """
     TODO : numbers need to be in the same format (reduced/extended). Program doesn't
       check for this condition
    """
    if isinstance(x,BigInt):
        #if not self.is_redc:
          return ZFieldEl((self.bignum * x.bignum) % ZField.get_extended_p().as_long())
        #else:
        #  return ZFieldEl._mul_redc(self.get(),x.bignum)
    elif isinstance(x,int) or isinstance(x,long):
        #if not self.is_redc:
          return (self.bignum * x) % ZField.get()
        #else:
        #  return ZFieldEl._mul_redc(self.get(),x)

   def __pow__ (self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,int) or isinstance(x,long):
         if not self.is_redc:
           return pow(self.bignum,x,ZField.get())
         else:
             return ZFieldEl._pow_redc(self.get(),x)
     elif isinstance(x,BigInt):
         if not self.is_redc:
           return pow(self.bignum,x.bignum,ZField.get())
         else:
           return ZFieldEl._pow_redc(self.get(),x)
     else :
       assert True,"Invalid type"
   
   def __floordiv__ (self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,int) or isinstance(x,long):
       return (self.bignum // x) % ZField.get()
     elif isinstance(x,BigInt):
       return (self.bignum // x.bignum) % ZField.get()
     else :
       assert True,"Invalid type"

   def __truediv__ (self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,int) or isinstance(x,long):
       return (self.bignum // x) % ZField.get()
     elif isinstance(x,BigInt):
       return (self.bignum // x.bignum) % ZField.get()
     else :
       assert True,"Invalid type"

   def __mod__ (self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,int) or isinstance(x,long):
       return (self.bignum % x) % ZField.get()
     elif isinstance(x,BigInt):
       return (self.bignum % x.bignum) % ZField.get()
     else :
       assert True,"Invalid type"

   def __iadd__(self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,BigInt):
       self.bignum = (self.bignum + x.bignum) % ZField.get()
       return self
     elif isinstance(x,int) or isinstance(x,long):
       self.bignum = (self.bignum + x) % ZField.get()
       return self
     else :
       assert True,"Invalid type"

   def __isub__(self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
       check for this condition
     """
     if isinstance(x,BigInt):
       self.bignum = (self.bignum - x.bignum) % ZField.get()
       return self
     elif isinstance(x,int) or isinstance(x,long):
       self.bignum = (self.bignum - x) % ZField.get()
       return self
     else :
       assert True,"Invalid type"

   def inv(self):
       y = ZFieldEl.inv(self.bignum)
       self = BigInt(y)
       return self

   @staticmethod
   def inv(x):
     assert ZField.is_init()==False, "Prime not initialized"
     if isinstance(x,BigInt):
         x = x.bignum
     elif not isinstance(x,int)  and not isinstance(x,long)   :
         assert True, "Invalid type"

     mod = ZField.get_extended_p()
     # Based on a simplification of the extended Euclidean algorithm
     assert mod > 0 and 0 <= x < mod
     y = x
     x = mod
     a = 0
     b = 1
     while y != 0:
        a, b = b, a - x // y * b
        x, y = y, x % y
     if x == 1:
         return a % mod
     else:
        raise ValueError("Reciprocal does not exist")
   # Comparison operators
   # <, <=, >, >=, ==, !=

   def __lt__(self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,BigInt):
       return self.bignum < x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum < x % ZField.get()
     else :
       assert True,"Invalid type"

   def __le__(self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,BigInt):
       return self.bignum <= x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum <= x % ZField.get()
     else :
       assert True,"Invalid type"

   def __eq__(self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,BigInt):
       return self.bignum == x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum == x % ZField.get()
     else :
       assert True,"Invalid type"

   def __ne__(self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,BigInt):
       return self.bignum != x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum != x % ZField.get()
     else :
       assert True,"Invalid type"

   def __gt__(self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,BigInt):
       return self.bignum > x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum > x % ZField.get()
     else :
       assert True,"Invalid type"

   def __ge__(self,x):
     """
      TODO : numbers need to be in the same format (reduced/extended). Program doesn't
        check for this condition
     """
     if isinstance(x,BigInt):
       return self.bignum >= x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum >= x % ZField.get()
     else :
       assert True,"Invalid type"

   def random(self,maxn, minn=0):
     if isinstance(maxn,BigInt):
        bnmaxn = maxn.bignum 
     elif isinstance(maxn,int) or isinstance(maxn,long):
        bnmaxn = maxn
     else :
       assert True,"Invalid type"
    
     if isinstance(minn,BigInt):
        bnminn = minn.bignum
     elif isinstance(minn,int) or isinstance(minn,long):
        bnminn = minn
     else :
       assert True,"Invalid type"

     return randint(minn,maxn) % ZField.get()

   def show (self):
       n = self.bignum
       print [n, ZField.get()]

   @staticmethod
   def _mul_redc(x,y):
       mod = ZField.get()
       assert 0 <= x < mod and 0 <= y < mod
       product = x * y
       temp = ((product & ZField.redc_data['Rmask']) * ZField.redc_data['Pp']) & ZField.redc_data['Rmask']
       reduced = (product + temp * mod) >> ZField.redc_data['bitlen']
       result = reduced if (reduced < mod) else (reduced - mod)
       assert 0 <= result < mod
       return result

   @staticmethod
   def _pow_redc(x,y):
       assert 0 <= x < ZField.get()
       if y < 0:
           raise ValueError("Negative exponent")
       z = ZField.redc_data['convertedone']
       while y != 0:
           if y & 1 != 0:
               z = ZFieldEl._mul_redc(z,x)
           x = ZFieldEl._mul_redc(x,x)
           y >>= 1
       return z
