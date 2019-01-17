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
//    - ZField only makes sense if initialized with prime.
//    
// ------------------------------------------------------------------

"""
from bigint import BigInt
import math
import numpy as np

class ZField:
  MAXP = long(1e10)
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
  prime = 0             # Current prime
  prime_ext = prime     # Extended prime
  roots = []
  inv_roots = []

  # Montgomery reduction data
  redc_data = {'init': False, 'bits' : 0, 'reducer' : 0, 'mask' : 0, 'reciprocal' : 0, 'factor' : 0, 'convertedone' : 0}

  # Factorization data
  factor_data = {'factors' :   [],  # prime factors of prime - 1
                'exponents' : []   # prime factors exponents of prime  - 1
               }

  def __init__(self,q,factor_data=None):
      self.set(q)
      ZField._factorize(factor_data)

  @classmethod
  def get(self):
    return ZField.prime.bignum

  @classmethod
  def get_extended(self):
    return ZField.prime_ext.bignum

  @classmethod
  def set(self,q):
    ZField.prime = BigInt(q)
    ZField.prime_ext = ZField.prime
    ZField.init_prime = True
    ZField.redc_data['init'] = False

  @classmethod
  def is_init(self):
      return ZField.init_prime

  @classmethod
  def is_redc(self):
      return ZField.redc_data['init']

  @classmethod
  def show(self):
      ZField.prime.show()

  @classmethod
  def reduce(self):
      if not ZField.redc_data['init']:
          bitlen = int(math.ceil(math.log(ZField.get(),2)))
          ZField.prime = BigInt(1 << bitlen | 1) # force it to be odd
          ZField.redc_data['bits'] = (ZField.get().bit_length() // 8 +1 )* 8 # Multiple of 8
          ZField.redc_data['reducer'] = 1 << ZField.redc_data['bits']
          ZField.redc_data['mask'] = ZField.redc_data['reducer'] - 1
          ZField.redc_data['reciprocal'] = ZFieldEl._inv(ZField.redc_data['reducer'] % ZField.get())
          ZField.redc_data['factor'] = (ZField.redc_data['reducer'] * ZField.redc_data['reciprocal'] - 1) // ZField.get()
          ZField.redc_data['convertedone'] = ZField.redc_data['reducer'] % ZField.get()

      ZField.redc_data['init'] = True

  @classmethod
  def extend(self):
      if ZField.redc_data['init']:
         ZField.prime = ZField.prime_ext

      ZField.redc_data['init'] = False

  @classmethod
  def _find_generator(cls):
      """
      Returns an arbitrary generator of the multiplicative group of integers module p
        for ZField where gen  ^ (k*nroots) = 1 mod p. If p is prime, an answer must exist
      """
      gamma=long(1)
      prime = ZField.get_extended()
      for i in xrange(len(ZField.factor_data['factors'])):
           beta = long(1)
           prime_factor = ZField.factor_data['factors'][i]
           exponent = ZField.factor_data['exponents'][i]
              
           # alpha is random number between 0 and mod (inclusive)
           while beta == 1:
             alpha = random.randint(0,prime))
             beta = pow(alpha,prime-1/prime_factor,prime)

           gamma = gamma * pow(alpha,prime-1/(prime_factor**exponent),prime)
           gamma = gamma % (prime)
       
      return gamma

  @classmethod
  def _find_primitive_root(cls, nroots):
      """
        Returns primitive root such that root = gen ^ nroots % prime,
      """
      gen = ZField._find_generator()
      prime = ZField.get_extended()
      return = pow(gen, (prime - 1) // nroots, prime)

  @classmethod
  def get_roots(cls):
    return ZField.roots, ZField.inv_roots

  @classmethod
  def find_roots(cls, nroots, find_inv_roots=False):
      """
        Returns nroots
      """
      if not ZField.is_init():
        assert True, "Prime not initialized"

      # initialize roots
      ZField.roots = [ 1, ZField._find_primitive_root(nroots) ]
      ZField.inv_roots = []
      for i in xrange(nroots)-2:
         ZField.roots.append(ZField.roots[i] * ZField.roots[i+1])

      if find_inv_roots:
         ZField.inv_roots = [1]
         ZField.inv_roots[1:] =  map(ZFieldEl._inv, ZField.roots[1:] )

      return ZField.roots, ZField.inv_roots

  @classmethod
  def _factorize(cls, factor_data=None):
      """
       Factorizes prime - 1. Only works for small primes less thanb MAXP.
      """
      if isinstance(factor_data,dict) and
        'factors' in factor_data and 'exponents' in factor_data:
           ZField.factor_data = factor_data
      
      elif isinstance(factor_data,str):
        if factor_data == ZField.BN128_DATA['curve']:
           ZField.factor_data = ZField.BN128_DATA['factor_data']
        elif factor_data == ZField.P1009_DATA['curve']:
           ZField.factor_data = ZField.P1009_DATA['factor_data']
        else:
          assert True, "Curve information not available"
          
      elif ZField.get_extended() > ZField.MAXP 
        assert True,"Prime is too large to factorize"a

      else:
        prime_1 = ZField.get_extended() - 1
        ZField.factor_data = ZField._prime_factors(prime_1)
            
  @classmethod
  def __prime_factors(cls,n):
     """
      Factorizes p_1 into prime factors. Returns dictionary with following info:
       'factors' : array of prime factors
       'exponents' : array of prime factor exponent
     """
     if n < 1:
         assert True,  "Number needs to be larger than 1"
     result = {'factors' :[], 'exponents' : [] }
     i = 2
     end = math.sqrt(n)
     while i <= end:
         if n % i == 0:
             n //= i
             result['factors'].append(i)
             while n % i == 0:
                 n //= i
             end = math.sqrt(n)
         i += 1
     if n > 1:
         result['factors'].append(n)
     
     factor_set = set(resut['factors'])
     for f in result['factors']:
       el_set = set([f])
       rem_factor_list = list(factor_set - el_set)
       rem_factor = np.prod(np.asarray(rem_factor))
       result['exponents'].append(math.log(n / rem_factor,f))
 
    if len(result['factors']) == 0:
      assert True, "Factorization could not find any prime factor"

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
         self.bignum = BigInt(bignum) % ZField.get_extended()
         if ZField.is_redc():
           self.reduce()
           self.is_redc = True
         else:
             self.is_redc = False
     else :
       assert True, "Prime not initialized"

   def reduce(self):
       if not self.is_redc:
           self.bignum = (self.bignum << ZField.redc_data['bits']) % ZField.get_extended()
           self.is_redc = True

   def extend(self):
       if self.is_redc:
           self.bignum = (self.bignum * ZField.redc_data['reciprocal']) % ZField.get()
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
      return (self.bignum - x) % Zfield.get()

   def __mul__(self,x):
    """
     TODO : numbers need to be in the same format (reduced/extended). Program doesn't
       check for this condition
    """
    if isinstance(x,BigInt):
        if not self.is_redc:
          return (self.bignum + x.bignum) % ZField.get()
        else:
          return ZFieldEl._mul_redc(self.get(),x.bignum)
    elif isinstance(x,int) or isinstance(x,long):
        if not self.is_redc:
          return (self.bignum * x) % ZField.get()
        else:
          return ZFieldEl._mul_redc(self.get(),x)

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
       y = ZFieldEl._inv(self.bignum)
       self = BigInt(y)
       return self

   @staticmethod
   def _inv(x):
     assert(ZField.is_init()==False, "Prime not initialized")
     if isinstance(x,BigInt):
         x = x.bignum
     elif not isinstance(x,int)  and not isinstance(x,long)   :
         assert True, "Invalid type"

     mod = ZField.get_extended()
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
       temp = ((product & ZField.redc_data['mask']) * ZField.redc_data['factor']) & ZField.redc_data['mask']
       reduced = (product + temp * mod) >> ZField.redc_data['bits']
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

if __name__ == '__main__':
    ZField("0x12345a12345678902345")
    ZField.show()
    x = ZFieldEl("0x12345a123456789023452345")
    x.show()
    y = BigInt("0x12345a145678902345")
    y.show()
    z = 4

    print "Sum"
    print x+y
    print "Sub"
    print x-y
    print "Mul"
    print x*y
    print "Pow"
    print x**z
    print "Div 2"
    print x//y
    #print "Div 1"
    #print x/y
    print "Mod"
    print x % z

    print "Inv 1"
    print x.inv().get()
    print x * x.inv().get()

    print "+="
    x += y
    x.show()
    print "-="
    x -= y
    x.show()

    print "Shift R"
    print x >> z

    print "Shft L"
    print x << z

    # Reduce
    x.show()
    print ZField.get()
    ZField.reduce()
    x.reduce()
    x.show()
    x.extend()
    x.show()
    x.reduce()

    print "Mul"
    print x*y

    print "Pow"
    print x**z
