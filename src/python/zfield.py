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
//   Implementation of finite field functionality and arithmetic. Includes three major classes:
//    ZField(BigInt) : Defintion of Finite field with prime characteristic Z/pZ
//    ZFieldEl(BigInt) : Class defining  Finite Field Elements. Defines general arithmetic
//      functionality and comparison functons
//    ZFieldElExt(ZFieldEl) : Class defining standard Finite field elements (non-reduced)
//    ZFieldElRedc(ZFieldEl) : Class defining Montgomert reduced Finity field elements
//   
//   ZField is inherited from BigInt. It must be initialized (only with a prime makes sense, although it is not 
//        verified) before finite field arithmetic can start. Field is initialized passing a prime number
//        (hex string, dec string, BigInt, long, int) and some optional factorization data required
//        if prime is beyond PRIME_THR. Upon initialization, Montgomery reduction is initialized automatically
//        so that operations can be performed either in Montgomery or extended domain.
//        Once Finite field is initialized, the n-th root of unity (and their inverses) can be computed
//
//       Methods:
//          Constructor __init__(self, q, factor_data=None)
//          @classmethod get_extended_p(cls) : Returns prime (extended) as BigInt
//          @classmethod get_reduced_p(cls) :  Returns prme (reduced) as BigInt
//          @classmethod get_reduction_data(cls) :  Returns reduce data in a dictionary
//          @classmethod is_init(cls) : Returns True if Finitize field is initialized
//          @classmethod reduce(cls) : Performs reduction operation. Initalizes and returns
//                 reduction data
//          @staticmethod reduce(x) : Inverts x (BigInt/int/long). Raises assertion
//                 if inversion is not possible
//          @staticmethod find_generator(cls): Returns arbitrary generator for Finite field
//          @classmethod find_primitive_root(cls, n): Computes and returs first n-th root of unity
//          @classmethod find_roots(cls, n, find_inv_roots=True): Computes, stores and returs 
//                   all n-th root of unity. If find_inv_roots is True, it also computes inverse roots
//          @classmethod get_roots(cls): Returns stored (via find_roots) roots
//          @classmethod factorize(cls, factor_data=None) : Factorizes Finite field prime - 1 and initializes
//                     necessary data structures to compute generator and n-th roots of unity. If prime
//                     is larger than PRIME_THE, factorization data needs to be provied. Factorization
//                     is automatically done when Finite Field is initialized
//          @classmethod get_factors(cls) : Returns factorization data of current Finite field
//
//   ZFieldEl is inherited from BigInt. Objects of this type can be isntantiated after ZField has been created.
//      ZFieldEl defines general Finite Field Element functionality including arithmetic, comparison and
//      bitwise operatos:
//       
//        Constructor : __init__(self,bignum)
//        same_format(self,P1)  ;   True if self and X are same format if Finite Field represenation (both extended or both reduced)
//        Arithmetic operators : +, -, neg (-x), +=, -=, % -> all aritmetic defined modulo p
//        Comparison operators : <, <=, >, >=, ==, !=
//        Bitwise operators    : <<, >>, <<=, >>=, &, |
//
//
//   ZfieldElExt is inherited from ZFieldEl. It defines the following functions of Finite Field elements
//      represented in normal (non Montgomery) format
//        Constructor : __init__(self,bignum)
//        reduce(self) : Returns ZFieldElRedc (Montgomery) representation of Finite Field element
//        Arithmetic operators : *, /, //,  pow -> all aritmetic is defined modulo p
//
//
//   ZfieldElRedc is inherited from ZFieldEl. It defines the following functions of Finite Field elements
//      represent in Montgomery format
//        Constructor : __init__(self,bignum)
//        extend(self) : Returns ZFieldElExt ( non Montgomery) representation of Finite Field element
//        Arithmetic operators : *, /, //,  pow -> all aritmetic is defined modulo p
//
//   TODO
//    - ZField only makes sense if initialized with prime.. Not checked
//    -  There is a multiplication of extended * reduced that i need to implement. For now, it is not alloed
//    - Montgomery / and //
// ------------------------------------------------------------------

"""
import math
import numpy as np
from random import randint
from abc import ABCMeta, abstractmethod

from bigint import BigInt
from zutils import *

class ZField(object):
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
  redc_data = {'Rbitlen' : 0, 'R' : 0, 'Rmask' : 0, 'Rp' : 0, 'Pp' : 0, 'convertedone' : 0}

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
  def get_reduction_data(cls):
    """
      Returns reduce data in a dictionary. Reduction operation for a Finite field with characteristic P
       is finding R, Rp and Pp such that 
          R * Rp = 1 (mod P)
          R * Rp - P * Pp = 1
       For reduction to be useful, P is a prime, and R is a power of 2 
      
       Dictionary Keys:
           Rbitlen :  (int) Number of bits in R
           R :        (long) R
           Rmask :    (long) (1 << R) - 1
           Rp :       (long) Rp
           Pp :       (long) Pp
           convertedone : ???
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
      if ZField.is_init():
        p = ZField.get_extended_p().as_long()
        bitlen = int(math.ceil(math.log(p,2)))
        t = 1 << bitlen | 1 # force it to be odd
        ZField.redc_data['Rbitlen'] = (t.bit_length() // 8 +1 )* 8 # Multiple of 8
        ZField.redc_data['R'] = long(1 << ZField.redc_data['Rbitlen'])
        ZField.redc_data['Rmask'] = long(ZField.redc_data['R'] - 1)
        ZField.redc_data['Rp'] = ZField.inv(ZField.redc_data['R'] % p).as_long()
        ZField.redc_data['Pp'] = (ZField.redc_data['R'] * ZField.redc_data['Rp'] - 1) // p
        ZField.redc_data['convertedone'] = ZField.redc_data['R'] % p
        ZField.redc_prime = BigInt(ZField.redc_data['R'])

      else:
         assert True, "Finite field not initialized"
        

  @staticmethod
  def inv(x):
     """
       returns X' such that X * X' = 1 (mod p)
     """
     if not ZField.is_init():
         assert True, "Finite field not initialized"
     elif isinstance(x,BigInt): 
         x_l = x.bignum
     elif isinstance(x,int)  or isinstance(x,long)   :
         x_l = x
     else :
         assert True, "Invalid type"

     num = x_l
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
         return ZFieldElExt(a % p)
     else:
        print "X : {}, P : {}".format(num , p)
        raise ValueError("Reciprocal does not exist")

  @classmethod
  def find_generator(cls):
      """
      Returns an arbitrary generator gen of the multiplicative group ZField with characteristic p
        where gen  ^ p = 1 mod p. If p is prime, an answer must exist
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
       
      return ZFieldElExt(gamma)

  @classmethod
  def find_primitive_root(cls, nroots):
      """
        Returns primitive root such that root = gen ^ nroots % prime,
      """
      if not ZField.is_init():
         assert True, "Finite field not initialized"

      gen = ZField.find_generator().as_long()
      prime = ZField.get_extended_p().as_long()
      return ZFieldElExt(pow(gen, (prime - 1) // nroots, prime))

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
      ZField.roots = [ ZFieldElExt(1), ZField.find_primitive_root(nroots) ]

      ZField.inv_roots = []
      root_1 = ZField.roots[1]
      for i in xrange(nroots-2):
         ZField.roots.append(root_1 * ZField.roots[-1])

      if find_inv_roots:
         ZField.inv_roots = [ZFieldElExt(1)]
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
        ZField.factor_data = ZUtils.prime_factors(prime_1)
 
      return ZField.factor_data
     
  @classmethod
  def get_factors(cls):
      """
        Returns factorization data for current Finite field in a dictionary with following keys
          'factors' : array of prime factors
          'exponents' : array of prime factor exponent
      """
      return ZField.factor_data


class ZFieldEl(BigInt):

   __metaclass__ = ABCMeta

   def __init__(self, bignum):
     """
      Constructor

      Parameters
      -----------
       bignum : BigInt/ZFieldElRedcd/ZFieldElExt -> Initializes element
     """
     if not ZField.is_init():
       assert True, "Finite Field not initialized"

     BigInt.__init__(self,bignum)
   
   def same_format(self, X):
      """
       True if self and X are same format if Finite Field represenation (both extended or both reduced)
      """
      return type(self) == type(X)

   # Arithmetic operators
   # +, -, *, %, pow, neg, +=, -=
   def __add__(self,x):
    """
      X + Y (mod P) : Add operation is the same for extended and reduced representations. 
       Result of addition is of type self. Note that ZFieldElExt and ZFieldElRedc
       cannot be added toguether
    """
    if isinstance(x,BigInt):
        if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
           assert True,"Invalid type"
        else:
          newz =  (self.bignum + x.bignum) % ZField.get_extended_p().as_long()
    elif isinstance(x,int) or isinstance(x,long):
      newz = (self.bignum + x) % ZField.get_extended_p().as_long()
    else :
      assert True,"Invalid type"
   
    if isinstance(self,ZFieldElExt) or isinstance(self,BigInt):
       return ZFieldElExt(newz)
    else :
       return ZFieldElRedc(newz)

   def __sub__(self,x):
    """
      X - Y (mod P): Sub operation is the same for extended and reduced representations
       Result of operation is of type self. Note that ZFieldElExt and ZFieldElRedc
       cannot be substracted toguether
    """
    if isinstance(x,BigInt):
        if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
           assert True,"Invalid type"
        else:
           newz = (self.bignum - x.bignum) % ZField.get_extended_p().as_long()
    elif isinstance(x,int) or isinstance(x,long):
      newz = (self.bignum - x) % ZField.get_extended_p().as_long()
    else :
      assert True,"Invalid type"

    if isinstance(self,ZFieldElExt):
       return ZFieldElExt(newz)
    else :
       return ZFieldElRedc(newz)

   def __neg__ (self):
     """
      -X (mod P)
       Result of negationaddition is of type self
     """
     newz = ZField.get_extended_p() - self.as_long()
     if isinstance(self,ZFieldElExt):
       return ZFieldElExt(newz)
     else :
       return ZFieldElRedc(newz)

   def __iadd__(self,x):
    """
      X = X + Y (mod P) : Add operation is the same for extended and reduced representations. 
       Result of addition is of type self. Note that ZFieldElExt and ZFieldElRedc
       cannot be added toguether
    """
    if isinstance(x,BigInt):
        if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
           assert True,"Invalid type"
        else:
           newz =  (self.bignum + x.bignum) % ZField.get_extended_p().as_long()
    elif isinstance(x,int) or isinstance(x,long):
      newz = (self.bignum + x) % ZField.get_extended_p().as_long()
    else :
      assert True,"Invalid type"
     
    if isinstance(self,ZFieldElExt):
       self = ZFieldElExt(newz)
    else :
       self =  ZFieldElRedc(newz)

    return self

   def __isub__(self,x):
    """
      X = X - Y (mod P) : Add operation is the same for extended and reduced representations. 
       Result of operation is of type self. Note that ZFieldElExt and ZFieldElRedc
       cannot be substracted toguether
    """
    if isinstance(x,BigInt):
        if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
           assert True,"Invalid type"
        else:
           newz =  (self.bignum - x.bignum) % ZField.get_extended_p().as_long()
    elif isinstance(x,int) or isinstance(x,long):
      newz = (self.bignum - x) % ZField.get_extended_p().as_long()
    else :
      assert True,"Invalid type"

    if isinstance(self,ZFieldElExt):
       self =  ZFieldElExt(newz)
    else :
       self =  ZFieldElRedc(newz)

    return self

   def __mod__(self,x):
    """
      X % Y (mod P) : Sub operation is the same for extended and reduced representations
       Result of mod is of type self. Note that ZFieldElExt and ZFieldElRedc
       cannot be mod toguether
    """
    if isinstance(x,BigInt):
        if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
           assert True,"Invalid type"
        else:
            newz = (self.bignum % x.bignum) % ZField.get_extended_p().as_long()

    elif isinstance(x,int) or isinstance(x,long):
      newz = (self.bignum % x) % ZField.get_extended_p().as_long()
    else :
      assert True,"Invalid type"

    if isinstance(self,ZFieldElExt):
       return ZFieldElExt(newz)
    else :
       return ZFieldElRedc(newz)

   @abstractmethod
   """
     X * Y (mod P) : Defined in child classes
   """
   def __mul__(self,x):
    pass

   @abstractmethod
   """
     X ** Y (mod P) : Defined in child classes
   """
   def __pow__ (self,x):
     pass
  
   @abstractmethod
   """
     X // Y (mod P) : Defined in child classes
   """
   def __floordiv__ (self,x):
     pass

   @abstractmethod
   """
     X / Y (mod P) : Defined in child classes
   """
   def __truediv__ (self,x):
     pass

 
   # Comparison operators
   # <, <=, >, >=, ==, !=
   def __lt__(self,x):
     """
      True if X < y
       
      NOTE : if x is a BigInt/int/long, i am comparing finite field element to int.
       if x is a FieldElement, it must be of the same type as self
     """
     if isinstance(x,BigInt):
         if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
              (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
            assert True,"Invalid type"
         else :
            return self.bignum < x.bignum % ZField.get_extended_p().as_long()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum < x % ZField.get_extended_p().as_long()
     else :
       assert True,"Invalid type"

   def __le__(self,x):
     """
      True if X <= y
       
      NOTE : if x is a BigInt/int/long, i am comparing finite field element to int.
       if x is a FieldElement, it must be of the same type as self
     """
     if isinstance(x,BigInt):
         if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
              (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
            assert True,"Invalid type"
         else :
            return self.bignum <= x.bignum % ZField.get_extended_p().as_long()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum <= x % ZField.get_extended_p().as_long()
     else :
       assert True,"Invalid type"

   def __eq__(self,x):
     """
      True if X == y
       
      NOTE : if x is a BigInt/int/long, i am comparing finite field element to int.
       if x is a FieldElement, it must be of the same type as self
     """
     if isinstance(x,BigInt):
         if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
              (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
            assert True,"Invalid type"
         else :
            return self.bignum == x.bignum % ZField.get_extended_p().as_long()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum == x % ZField.get_extended_p().as_long()
     else :
       assert True,"Invalid type"

   def __ne__(self,x):
     """
      True if X != y
       
      NOTE : if x is a BigInt/int/long, i am comparing finite field element to int.
       if x is a FieldElement, it must be of the same type as self
     """
     if isinstance(x,BigInt):
         if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
              (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
            assert True,"Invalid type"
         else :
            return self.bignum != x.bignum % ZField.get_extended_p().as_long()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum != x % ZField.get_extended_p().as_long()
     else :
       assert True,"Invalid type"

   def __gt__(self,x):
     """
      True if X > y
       
      NOTE : if x is a BigInt/int/long, i am comparing finite field element to int.
       if x is a FieldElement, it must be of the same type as self
     """
     if isinstance(x,BigInt):
         if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
              (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
            assert True,"Invalid type"
         else :
            return self.bignum > x.bignum % ZField.get_extended_p().as_long()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum > x % ZField.get_extended_p().as_long()
     else :
       assert True,"Invalid type"

   def __ge__(self,x):
     """
      True if X >= y
       
      NOTE : if x is a BigInt/int/long, i am comparing finite field element to int.
       if x is a FieldElement, it must be of the same type as self
     """
     if isinstance(x,BigInt):
         if (isinstance(x,ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
             (isinstance(x,ZFieldElExt) and isinstance(self, ZFieldElRedc)):
            assert True,"Invalid type"
         else :
            return self.bignum >= x.bignum % ZField.get_extended_p().as_long()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum >= x % ZField.get_extended_p().as_long()
     else :
       assert True,"Invalid type"

   # Bitwise operators
   #  <<, >>, <<=, >>=, &, |
   def __lshift__(self,x):
     """
      X << x . Returns a BigInt type
        x : BigInt/int/long/ZFieldElExt

      NOTE : x cannot be ZfieldElRedc
     """
     if isinstance(x,BigInt):
         if isinstance(x,ZFieldElRedc) :
            assert True,"Invalid type"
         else :
            return BigInt(self.as_long() << x)
     elif isinstance(x,int) or isinstance(x,long):
            return BigInt(self.as_long() << x)
     else :
       assert True,"Invalid type"


   def __rshift__(self,x):
     """
      X >> x . Returns a BigInt type
        x : BigInt/int/long/ZFieldElExt

      NOTE : x cannot be ZfieldElRedc
     """
     if isinstance(x,BigInt):
         if isinstance(x,ZFieldElRedc): 
            assert True,"Invalid type"
         else :
            return BigInt(self.as_long() >> x)
     elif isinstance(x,int) or isinstance(x,long):
            return BigInt(self.as_long() >> x)
     else :
       assert True,"Invalid type"

   def __irshift__(self,x):
     """
      X = X << x 

        Note Unsupported operation
     """
     assert True, "Unsupported operation"

   def __ilshift__(self,x):
     """
      X << x .

      NOTE : Unsupported operation
     """
     assert True, "Unsupported operation"
  
   def __and__(self,x):
     """
      X & x . Returns a BigInt type
        x : BigInt/int/long/ZFieldElExt

      NOTE : x cannot be ZfieldElRedc
     """
     if isinstance(x,BigInt):
         if isinstance(x,ZFieldElRedc):
            assert True,"Invalid type"
         else :
            return BigInt(self.as_long() & x)
     elif isinstance(x,int) or isinstance(x,long):
            return BigInt(self.as_long() & x)
     else :
       assert True,"Invalid type"
   
   def __or__(self,x):
     """
      X | x . Returns a BigInt type
        x : BigInt/int/long/ZFieldElExt

      NOTE : x cannot be ZfieldElRedc
     """
     if isinstance(x,BigInt):
         if isinstance(x,ZFieldElRedc):
            assert True,"Invalid type"
         else :
            return BigInt(self.as_long() | x)
     elif isinstance(x,int) or isinstance(x,long):
            return BigInt(self.as_long() | x)
     else :
       assert True,"Invalid type"


class ZFieldElExt(ZFieldEl):

   def __init__(self, bignum):
     """
      Constructor

      Parameters
      -----------
       bignum : BigInt/ZFieldElRedc/ZFieldElExt -> Initializes element
     """
     
     if not ZField.is_init():
       assert True, "Finite Field not initialized"
   
     if not isinstance(bignum,BigInt) and \
          not isinstance(bignum,int)  and \
          not isinstance(bignum,long):
       assert True, "Incorrect Finite Field element format"

     ZFieldEl.__init__(self,bignum % ZField.get_extended_p().as_long())

   def reduce(self):
     """
      Performs Montgomery reduction operation. ZFieldElRedc object is returned
     """
     reduction_data = ZField.get_reduction_data()
     reduced_z =   ZFieldElRedc((self.as_long() << reduction_data['Rbitlen']) % ZField.get_extended_p().as_long())

     return reduced_z

   # Arithmetic operators
   # *, /, pow
   def __mul__(self,x):
    """
      X * Y (mod P) : returns ZFieldElExt
        Y can be BigInt/Int/long or ZFieldElExt

      TODO there is a multiplication of extended * reduced that i need to implement. For now, it is not allowed
    """
    if isinstance(x,BigInt):
        if isinstance(x,ZFieldElRedc):
           assert True,"Invalid type"
        else:
           return ZFieldElExt((self.bignum * x.bignum) % ZField.get_extended_p().as_long())
    elif isinstance(x,int) or isinstance(x,long):
        return ZFieldElExt(self.bignum * x) % ZField.get_extended_p().as_long()
    else :
       assert True,"Invalid type"

   def __pow__ (self,x):
     """
      X ** Y (mod P) : returns ZFieldElExt
        Y can be BigInt/Int/long or ZFieldElExt
     """
     if isinstance(x,int) or isinstance(x,long):
         return ZFieldElExt(pow(self.bignum,x,ZField.get_extended_p().as_long()))
     elif isinstance(x,BigInt):
         if isinstance(x,ZFieldElRedc):
            assert True,"Invalid type"
         else:
            return ZFieldElExt(pow(self.bignum,x.bignum,ZField.get_extended_p().as_long()))
     else :
       assert True,"Invalid type"
  
   def __floordiv__ (self,x):
     """
      X // Y (mod P) : returns ZFieldElExt
        Y can be BigInt/Int/long or ZFieldElExt
     """
     if isinstance(x,int) or isinstance(x,long):
       return ZFieldElExt((self.bignum // x) % ZField.get_extended_p().as_long())
     elif isinstance(x,BigInt):
         if isinstance(x,ZFieldElRedc):
            assert True,"Invalid type"
         else:
            return ZFieldElExt((self.bignum // x.bignum) % ZField.get_extended_p().as_long())
     else :
       assert True,"Invalid type"

   def __truediv__ (self,x):
     """
      X / Y (mod P) : returns ZFieldElExt
        Y can be BigInt/Int/long or ZFieldElExt

      NOTE : It doesn't work
     """
     if isinstance(x,int) or isinstance(x,long):
       return ZFieldElExt((self.bignum // x) % ZField.get_extended_p().as_long())
     elif isinstance(x,BigInt):
         if isinstance(x,ZFieldElRedc):
            assert True,"Invalid type"
         else:
            return ZFieldElExt((self.bignum // x.bignum) % ZField.get_extended_p().as_long())
     else :
       assert True,"Invalid type"

class ZFieldElRedc(ZFieldEl):

   def __init__(self, bignum):
     """
      Constructor

      Parameters
      -----------
       bignum : BigInt/ZFieldElRedc/ZFieldElExt -> Initializes element
     """
     if not ZField.is_init():
       assert True, "Finite Field not initialized"

     if not isinstance(bignum,BigInt) and  \
          not isinstance(bignum,int)  and  \
          not isinstance(bignum,long):
       assert True, "Incorrect Finite Field element format"

     ZFieldEl.__init__(self,bignum % ZField.get_extended_p().as_long())


   def extend(self):
     """
      Converts Montgomery representation to default. ZFieldElExt object is returned
     """
     reduction_data = ZField.get_reduction_data()
     extended_z =   ZFieldElExt((self.as_long() * reduction_data['Rp']) % ZField.get_extended_p().as_long())

     return extended_z

   # Arithmetic operators
   # * pow
   def __mul__(self,x):
    """
      X * Y (mod P) : returns ZFieldElRedc
        Y can be BigInt/Int/long or ZFieldElRedc
    """
      if isinstance(x,BigInt):
        x_l = x.as_long()
        if isinstance(x,ZFieldElExt):
           assert True,"Invalid type"
      elif isinstance(x,int) or isinstance(x,long):
          x_l = x
      else:
           assert True,"Invalid type"

      mod = ZField.get_extended_p().as_long()
      reduction_data = ZField.get_reduction_data()

      product = x_l * self.as_long()
      temp = ((product & reduction_data['Rmask']) * reduction_data['Pp']) & reduction_data['Rmask']
      reduced = (product + temp * mod) >> reduction_data['Rbitlen']
      result = reduced if (reduced < mod) else (reduced - mod)

      return ZFieldElRedc(result)

   def __pow__ (self,x):
     """
      X ** Y (mod P) : returns ZFieldElRedc
        Y can be BigInt/Int/long or ZFieldElExt
     """
     if isinstance(x,BigInt):
        x_l = x.as_long()
        if isinstance(x,ZFieldElExt):
           assert True,"Invalid type"
        elif x < 0:
           assert True,"Negative exponent"
     elif isinstance(x,int) or isinstance(x,long):
          x_l = x
     else:
           assert True,"Invalid type"

     tmpEl = ZFieldElRedc(self)
     z = ZField.get_reduction_data()['convertedone']
     while x_l != 0:
         if x_l & 1 != 0:
             z = (tmpEl * z).as_long()
         tmpEl = tmpEl * tmpEl
         x_l >>= 1
     return ZFieldElRedc(z)
  
   def __floordiv__ (self,x):
     """
       X // Y : TODO
     """
     assert True,"Operation not supported"

   def __truediv__ (self,x):
     """
      X / Y : TODO 
     """
     assert True,"Operation not supported"
