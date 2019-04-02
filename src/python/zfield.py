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
//          @staticmethod inv(x) : Inverts x (BigInt/int/long/ZFieldElExt). Raises assertion
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
//        Arithmetic operators : +, -, neg (-x), +=, -=, %-> all aritmetic defined modulo p
//        Comparison operators : <, <=, >, >=, ==, !=
//        Bitwise operators    : <<, >>, <<=, >>=, &, |
//            return ZFieldElExt/ZFieldElRedc depending on type
//
//
//   ZfieldElExt is inherited from ZFieldEl. It defines the following functions of Finite Field elements
//      represented in normal (non Montgomery) format
//        Constructor : __init__(self,bignum)
//        reduce(self) : Returns ZFieldElRedc (Montgomery) representation of Finite Field element
//        Arithmetic operators : *, //,  pow, inv -> all aritmetic is defined modulo p
//
//
//   ZfieldElRedc is inherited from ZFieldEl. It defines the following functions of Finite Field elements
//      represent in Montgomery format
//        Constructor : __init__(self,bignum)
//        extend(self) : Returns ZFieldElExt ( non Montgomery) representation of Finite Field element
//        Arithmetic operators : *,  //,  pow, inv -> all aritmetic is defined modulo p
//
//   TODO
//    - ZField only makes sense if initialized with prime.. Not checked
//    -  There is a multiplication of extended * reduced that i need to implement. For now, it is not alloed
// ------------------------------------------------------------------

"""
import math
from abc import ABCMeta, abstractmethod
from random import randint

import numpy as np
from bigint import BigInt
from zutils import *

class ZField(object):
    PRIME_THR = long(1e10)

    init_prime = False  # Flag : Is prime initialized
    ext_prime = []  # Field prime
    redc_prime = []  # Field Prime (montgomery reduced)
    roots = []  # Field roots of unity
    inv_roots = []  # Filed inverse roots of unity

    # Montgomery reduction data
    redc_data = []

    # Factorization data
    factor_data = []
    #factor_data.append({'factors': [],  # prime factors of prime - 1
    #               'exponents': []  # prime factors exponents of prime  - 1
    #                   })

    active_prime_idx = 0

    def __init__(self, q, factor_data=None):
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
        ZField.ext_prime = [BigInt(q)]
        ZField.redc_prime = []
        ZField.roots = []
        ZField.inv_roots = []
        ZField.redc_data= []
        ZField.factor_data = []
        ZField.active_prime_idx = 0
        # Apply processing for future reduction
        ZField.reduce()
        # init factorization data
        ZField.factorize(factor_data=factor_data)
        ZField.add_roots([],[])

    @classmethod
    def set_field(cls,idx):
        ZField.active_prime_idx=idx

    @classmethod
    def get_field(cls):
        return ZField.active_prime_idx

    @classmethod
    def add_field(cls,p, factor_data=None):
        ZField.ext_prime.append(BigInt(p))
        idx = len(ZField.ext_prime)-1
        old_idx = ZField.active_prime_idx
        ZField.active_prime_idx = idx
        ZField.reduce()
        ZField.factorize(factor_data=factor_data)
        ZField.add_roots([],[])
        ZField.active_prime_idx = old_idx

    @classmethod
    def add_roots(cls, r,ri):
        ZField.roots.append(r)
        ZField.inv_roots.append(ri)

    @classmethod
    def get_extended_p(cls):
        """
          Returns extended prime as BigInt
        """
        if not ZField.is_init():
            return None
        idx = ZField.active_prime_idx
        return ZField.ext_prime[idx]

    @classmethod
    def get_reduced_p(cls):
        """
          Returns reduced prime as BigInt.
           If None, it means that Field has't undergone Montgomery reduction
        """
        if not ZField.is_init():
            return None
        idx = ZField.active_prime_idx
        return ZField.redc_prime[idx]

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
               RmodP :    (long) R (mod P)
               R3modP :   (long) R^3 (mod P)
        """
        idx = ZField.active_prime_idx
        return ZField.redc_data[idx]

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

            NOTE : function based on https://www.nayuki.io/page/montgomery-reduction-algorithm
        """
        if ZField.is_init():
            redc_data = {'Rbitlen': 0, 'R': 0, 'Rmask': 0, 'Rp': 0, 'Pp': 0, 'RmodP': 0, 'R3modP': 0}
            p = ZField.get_extended_p().as_long()
            bitlen = int(math.ceil(math.log(p, 2)))
            t = 1 << bitlen | 1  # force it to be odd
            redc_data['Rbitlen'] = (t.bit_length() // 8 + 1) * 8  # Multiple of 8
            redc_data['R'] = long(1 << redc_data['Rbitlen'])
            redc_data['Rmask'] = long(redc_data['R'] - 1)
            redc_data['Rp'] = ZField.inv(redc_data['R'] % p).as_long()
            redc_data['Pp'] = (redc_data['R'] * redc_data['Rp'] - 1) // p
            redc_data['RmodP'] = redc_data['R'] % p
            redc_data['R3modP'] = (redc_data['R'] * redc_data['R'] * redc_data['R']) % p

            ZField.redc_prime.append(BigInt(redc_data['R']))
            ZField.redc_data.append(redc_data)

        else:
            assert False, "Finite field not initialized"

    @staticmethod
    def inv(x):
        """
          returns X' such that X * X' = 1 (mod p).
          X cannot be in Montgomery format

          NOTE : function based on https://www.nayuki.io/page/montgomery-reduction-algorithm
        """
        if not ZField.is_init():
            assert False, "Finite field not initialized"
        elif isinstance(x, ZFieldElRedc):
            assert False, "Invalid type"
        elif isinstance(x, BigInt):
            x_l = x.bignum
        elif isinstance(x, int) or isinstance(x, long):
            x_l = x
        else:
            assert False, "Invalid type"

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
            print "X : {}, P : {}".format(num, p)
            assert False, "Reciprocal does not exist"

    @classmethod
    def find_generator(cls):
        """
        Returns an arbitrary generator gen of the multiplicative group ZField with characteristic p
          where gen  ^ p = 1 mod p. If p is prime, an answer must exist

        NOTE : algorithm based on A Computational Introduction to Number Theory and Algebra by Victor Shoup,
           Section 11.1 (page 328)
        """
        if not ZField.is_init():
            assert False, "Finite field not initialized"

        alpha = long(100)
        idx = ZField.active_prime_idx
        gamma = long(1)
        prime = ZField.get_extended_p().as_long()
        for i in xrange(len(ZField.factor_data[idx]['factors'])):
            beta = long(1)
            prime_factor = ZField.factor_data[idx]['factors'][i]
            exponent = ZField.factor_data[idx]['exponents'][i]

            # alpha is random number between 0 and mod (inclusive)
            while beta == 1:
                #alpha = randint(1, prime-1)
                alpha +=1
                beta = pow(alpha, (prime - 1) / prime_factor, prime)

            gamma = gamma * pow(alpha, (prime - 1) / pow(prime_factor, exponent,prime), prime)
            gamma = gamma % (prime)

        return ZFieldElExt(gamma)

    @classmethod
    def find_primitive_root(cls, nroots):
        """
          Returns primitive root such that root = gen ^ nroots % prime,

         NOTE : function based on https://www.nayuki.io/page/montgomery-reduction-algorithm
        """
        if not ZField.is_init():
            assert False, "Finite field not initialized"

        gen = ZField.find_generator().as_long()
        prime = ZField.get_extended_p().as_long()
        return ZFieldElExt(pow(gen, (prime - 1) // nroots, prime))

    @classmethod
    def get_roots(cls):
        """
          returns computed roots of unity
        """
        if not ZField.is_init():
            return ([],[])
        idx = ZField.active_prime_idx
        return ZField.roots[idx], ZField.inv_roots[idx]

    @classmethod
    def find_roots(cls, nroots, find_inv_roots=True, rformat_ext=True):
        """
          Computes and returns nroots of unity. If find_inv_roots is True, inverse roots are
            also computed

            TODO : roots are [1, r1, r2, r3,..., rn-1], inv_roots = [1,rn-1, rn-2,... r2, r1].
             So, it is not necessary to store both versions. We can just store half
        """
        if not ZField.is_init():
            assert False, "Prime not initialized"

        idx = ZField.active_prime_idx
        # initialize roots
        ZField.roots[idx] = [ZFieldElExt(1), ZField.find_primitive_root(nroots)]

        root_1 = ZField.roots[idx][1]
        for i in xrange(nroots - 2):
            ZField.roots[idx] = ZField.roots[idx] + [root_1 * ZField.roots[idx][-1]]

        if find_inv_roots:
            ZField.inv_roots[idx] = [ZFieldElExt(1)]
            ZField.inv_roots[idx][1:] = map(ZField.inv, ZField.roots[idx][1:])

        if rformat_ext == False:
            ZField.roots[idx] = [x.reduce() for x in ZField.roots[idx]]
            ZField.inv_roots[idx] = [x.reduce() for x in ZField.inv_roots[idx]]

        return ZField.roots[idx], ZField.inv_roots[idx]

    @classmethod
    def factorize(cls, factor_data=None):
        """
         Factorizes prime - 1. Only works for small primes less than PRIME_THR.
        """
        if not ZField.is_init():
            assert False, "Finite field not initialized"
        elif isinstance(factor_data, dict) and 'factors' in factor_data and 'exponents' in factor_data:
            ZField.factor_data.append(factor_data)

        elif isinstance(factor_data, str):
            if factor_data in ZUtils.CURVE_DATA:
                ZField.factor_data.append(ZUtils.CURVE_DATA[factor_data]['factor_data'])
            else:
                assert False, "Field information not available"

        elif ZField.get_extended_p() > ZField.PRIME_THR:
            ZField.factor_data.append([None])

        else:
            prime_1 = ZField.get_extended_p().as_long() - 1
            ZField.factor_data.append(ZUtils.prime_factors(prime_1))

        return ZField.factor_data[-1]

    @classmethod
    def get_factors(cls):
        """
          Returns factorization data for current Finite field in a dictionary with following keys
            'factors' : array of prime factors
            'exponents' : array of prime factor exponent
        """
        if not ZField.is_init():
            return {'factors' : [], 'exponents' : []} 
        idx = ZField.active_prime_idx
        return ZField.factor_data[idx]


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
            assert False, "Finite Field not initialized"
        elif bignum is None:
            self.bignum = None
        else:
            BigInt.__init__(self, bignum)

    def as_list(self):
        return [self.bignum]

    # Arithmetic operators
    # +, -, *, %, pow, neg, +=, -=
    def __add__(self, x):
        """
          X + Y (mod P) : Add operation is the same for extended and reduced representations.
           Result of addition is of type self. Note that ZFieldElExt and ZFieldElRedc
           cannot be added toguether
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                newz = (self.bignum + x.bignum)
        elif isinstance(x, int) or isinstance(x, long):
            newz = (self.bignum + x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElRedc):
            return ZFieldElRedc(newz)
        else:
            return ZFieldElExt(newz)

    def __sub__(self, x):
        """
          X - Y (mod P): Sub operation is the same for extended and reduced representations
           Result of operation is of type self. Note that ZFieldElExt and ZFieldElRedc
           cannot be substracted toguether
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                newz = (self.bignum - x.bignum)
        elif isinstance(x, int) or isinstance(x, long):
            newz = (self.bignum - x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElRedc):
            return ZFieldElRedc(newz)
        else:
            return ZFieldElExt(newz)

    def __neg__(self):
        """
         -X (mod P)
          Result of negationaddition is of type self
        """
        newz = ZField.get_extended_p() - self.as_long()
        if isinstance(self, ZFieldElRedc):
            return ZFieldElRedc(newz)
        else:
            return ZFieldElExt(newz)

    def __iadd__(self, x):
        """
          X = X + Y (mod P) : Add operation is the same for extended and reduced representations.
           Result of addition is of type self. Note that ZFieldElExt and ZFieldElRedc
           cannot be added toguether
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                #newz = (self.bignum + x.bignum)
                newz = self + x
        elif isinstance(x, int) or isinstance(x, long):
            newz = (self.bignum + x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElRedc):
            self = ZFieldElRedc(newz)
        else:
            self = ZFieldElExt(newz)

        return self

    def __isub__(self, x):
        """
          X = X - Y (mod P) : Add operation is the same for extended and reduced representations.
           Result of operation is of type self. Note that ZFieldElExt and ZFieldElRedc
           cannot be substracted toguether
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                newz = self - x
                #newz = (self.bignum - x.bignum)
        elif isinstance(x, int) or isinstance(x, long):
            newz = (self.bignum - x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElRedc):
            self = ZFieldElRedc(newz)
        else:
            self = ZFieldElExt(newz)

        return self

    def __mod__(self, x):
        """
          X % Y (mod P) : Sub operation is the same for extended and reduced representations
           Result of mod is of type self. Note that ZFieldElExt and ZFieldElRedc
           cannot be mod toguether
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                newz = (self.bignum % x.bignum)

        elif isinstance(x, int) or isinstance(x, long):
            newz = (self.bignum % x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElRedc):
            return ZFieldElRedc(newz)
        else:
            return ZFieldElExt(newz)

    @abstractmethod
    def reduce(self):
        pass

    @abstractmethod
    def extend(self):
        pass

    @abstractmethod
    def __mul__(self, x):
        """
        X * Y (mod P) : Defined in child classes
        """
        pass

    @abstractmethod
    def __pow__(self, x):
        """
         X ** Y (mod P) : Defined in child classes
        """
        pass

    @abstractmethod
    def __div__(self, x):
        """
         X / Y (mod P) : Defined in child classes
        """
        pass

    @abstractmethod
    def inv(self):
        """
         Compute inverse of X
        """
        pass

    # Comparison operators
    # <, <=, >, >=, ==, !=
    def __lt__(self, x):
        """
         True if X < y

         NOTE : if x is a FieldElement, it must be of the same type as self.
          Comparing Field Element with a BigInt/int/long. Function looks at actual numbers
          as if they were signed integers
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                return self.bignum < x.bignum
        elif isinstance(x, int) or isinstance(x, long):
            return self.bignum < x
        else:
            assert False, "Invalid type"

    def __le__(self, x):
        """
         True if X <= y

         NOTE : if x is a FieldElement, it must be of the same type as self.
          Comparing Field Element with a BigInt/int/long. Function looks at actual numbers
          as if they were signed integers
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                return self.bignum <= x.bignum
        elif isinstance(x, int) or isinstance(x, long):
            return self.bignum <= x
        else:
            assert False, "Invalid type"

    def __eq__(self, x):
        """
         True if X == y

         NOTE : if x is a FieldElement, it must be of the same type as self.
          Comparing Field Element with a BigInt/int/long. Function looks at actual numbers
          as if they were signed integers
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                return self.bignum == x.bignum
        elif isinstance(x, int) or isinstance(x, long):
            return self.bignum == x
        else:
            assert False, "Invalid type"

    def __ne__(self, x):
        """
         True if X != y

         NOTE : if x is a FieldElement, it must be of the same type as self.
          Comparing Field Element with a BigInt/int/long. Function looks at actual numbers
          as if they were signed integers
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                return self.bignum != x.bignum
        elif isinstance(x, int) or isinstance(x, long):
            return self.bignum != x
        else:
            assert False, "Invalid type"

    def __gt__(self, x):
        """
         True if X > y

         NOTE : if x is a FieldElement, it must be of the same type as self.
          Comparing Field Element with a BigInt/int/long. Function looks at actual numbers
          as if they were signed integers
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                return self.bignum > x.bignum
        elif isinstance(x, int) or isinstance(x, long):
            return self.bignum > x
        else:
            assert False, "Invalid type"

    def __ge__(self, x):
        """
         True if X >= y

         NOTE : if x is a FieldElement, it must be of the same type as self.
          Comparing Field Element with a BigInt/int/long. Function looks at actual numbers
          as if they were signed integers
        """
        if isinstance(x, BigInt):
            if (isinstance(x, ZFieldElRedc) and isinstance(self, ZFieldElExt)) or \
                    (isinstance(x, ZFieldElExt) and isinstance(self, ZFieldElRedc)):
                assert False, "Invalid type"
            else:
                return self.bignum >= x.bignum
        elif isinstance(x, int) or isinstance(x, long):
            return self.bignum >= x
        else:
            assert False, "Invalid type"

    # Bitwise operators
    #  <<, >>, <<=, >>=, &, |
    def __lshift__(self, x):
        """
         X << x . Returns a ZfieldEl
           x : BigInt/int/long/ZFieldElExt
        """
        if isinstance(x, ZFieldElRedc):
            assert False, "Invalid type"
        elif isinstance(x, BigInt):
            newZ = (self.as_long() << x.as_long())
        elif isinstance(x, int) or isinstance(x, long):
            if isinstance(x, Z2FieldEl):
               newZ = [self.P[0].as_long() << x, self.P[1].as_long() << x]
            else :
               newZ = (self.as_long() << x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElExt):
            return ZFieldElExt(newZ)
        else:
            return ZFieldElRedc(newZ)

    def __rshift__(self, x):
        """
         X >> x . Returns a BigInt type
           x : BigInt/int/long/ZFieldElExt

         NOTE : x cannot be ZfieldElRedc
        """
        if isinstance(x, ZFieldElRedc):
            assert False, "Invalid type"
        elif isinstance(x, BigInt):
            newZ = (self.as_long() >> x.as_long())
        elif isinstance(x, int) or isinstance(x, long):
            newZ = (self.as_long() >> x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElExt):
            return ZFieldElExt(newZ)
        else:
            return ZFieldElRedc(newZ)

    def __irshift__(self, x):
        """
         X = X >> x
         NOTE : x cannot be ZfieldElRedc
        """
        if isinstance(x, ZFieldElRedc):
            assert False, "Invalid type"
        elif isinstance(x, BigInt):
            newZ = (self.as_long() >> x.as_long())
        elif isinstance(x, int) or isinstance(x, long):
            newZ = (self.as_long() >> x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElExt):
            self = ZFieldElExt(newZ)
        else:
            self = ZFieldElRedc(newZ)
        return self

    def __ilshift__(self, x):
        """
         X = X << x .

         NOTE : x cannot be ZfieldElRedc
        """
        if isinstance(x, ZFieldElRedc):
            assert False, "Invalid type"
        elif isinstance(x, BigInt):
            newZ = (self.as_long() << x.as_long())
        elif isinstance(x, int) or isinstance(x, long):
            newZ = (self.as_long() << x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElExt):
            self = ZFieldElExt(newZ)
        else:
            self = ZFieldElRedc(newZ)

        return self

    def __and__(self, x):
        """
         X & x . Returns a BigInt type
           x : BigInt/int/long/ZFieldElExt

         NOTE : x cannot be ZfieldElRedc
        """
        if isinstance(x, ZFieldElRedc):
            assert False, "Invalid type"
        elif isinstance(x, BigInt):
            newZ = (self.as_long() & x.as_long())
        elif isinstance(x, int) or isinstance(x, long):
            newZ = (self.as_long() & x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElExt):
            return ZFieldElExt(newZ)
        else:
            return ZFieldElRedc(newZ)

    def __or__(self, x):
        """
         X | x . Returns a BigInt type
           x : BigInt/int/long/ZFieldElExt

         NOTE : x cannot be ZfieldElRedc
        """
        if isinstance(x, ZFieldElRedc):
            assert False, "Invalid type"
        elif isinstance(x, BigInt):
            newZ = (self.as_long() | x.as_long())
        elif isinstance(x, int) or isinstance(x, long):
            newZ = (self.as_long() | x)
        else:
            assert False, "Invalid type"

        if isinstance(self, ZFieldElExt):
            return ZFieldElExt(newZ)
        else:
            return ZFieldElRedc(newZ)

    @classmethod
    def as_zel(self,l):
       if type(l) is list:
           if isinstance(l[0],ZfieldElExt) or isinstance(l[0],ZfieldElRedc):
              return l
           elif isinstance(l[0],int) or isinstance(l[0],long) or isinstance(l[0], BigInt):
              return [ZFieldElExt(t) for t in l]
           else:
              assert False, "Unexpected type"
       elif type(l) is dict:
           if isinstance(l.keys()[0],ZfieldElExt) or isinstance(l.keys()[0],ZfieldElRedc):
              return l
           elif isinstance(l.keys[0],int) or isinstance(l.keys()[0],long) or \
                  isinstance(l.keys[0], BigInt):
              return {i : ZFieldElExt(p[i]) for i in p.keys()}
           else:
              assert False, "Unexpected type"
       elif isinstance(l,ZfieldElExt) or isinstance(l,ZfieldElRedc):
           return l
       elif isinstance(l,int) or isinstance(l,long) or isinstance(l, BigInt):
           return ZFieldElExt(l)
       else:
            assert False, "Unexpected type"
            
         

class ZFieldElExt(ZFieldEl):

    def __init__(self, bignum):
        """
         Constructor

         Parameters
         -----------
          bignum : BigInt/ZFieldElRedc/ZFieldElExt -> Initializes element
        """

        if not ZField.is_init():
            assert False, "Finite Field not initialized"

        if bignum is None:
            ZFieldEl.__init__(None)
        elif not isinstance(bignum, BigInt) and \
                not isinstance(bignum, int) and \
                not isinstance(bignum, long):
            assert False, "Incorrect Finite Field element format"
        else:
            ZFieldEl.__init__(self, bignum % ZField.get_extended_p().as_long())

    def reduce(self):
        """
         Performs Montgomery reduction operation. ZFieldElRedc object is returned
        """
        reduction_data = ZField.get_reduction_data()
        reduced_z = ZFieldElRedc(self.as_long() << reduction_data['Rbitlen'])

        return reduced_z

    def extend(self):
        """
          Do nothing
        """
        return ZFieldElExt(self)

    # Arithmetic operators
    # *, /, pow
    def __mul__(self, x):
        """
          X * Y (mod P) : returns ZFieldElExt
            Y can be BigInt/Int/long or ZFieldElExt. IF Y is ZFieldExtRedc, call ZFieldExtRedc *

        """
        if isinstance(x, ZFieldElRedc):
            return x * self  # Montgomery multiplication
        elif isinstance(x, int) or isinstance(x, long):
            return ZFieldElExt((self.bignum * x))
        elif isinstance(x, BigInt):
            try:
               return ZFieldElExt((self.bignum * x.as_long()))
            except  AttributeError :
                return x * self
        else:
            return x * self

    def __rmul__(self, x):
        return self * x

    def __pow__(self, x):
        """
         X ** Y (mod P) : returns ZFieldElExt
           Y can be BigInt/Int/long or ZFieldElExt
        """
        if isinstance(x, int) or isinstance(x, long):
            return ZFieldElExt(pow(self.bignum, x))
        elif isinstance(x, BigInt):
            if isinstance(x, ZFieldElRedc):
                assert False, "Invalid type"
            else:
                return ZFieldElExt(pow(self.bignum, x.bignum))
        else:
            assert False, "Invalid type"

    def __div__(self, x):
        """
          X / Y (mod P) : returns ZFieldElExt
            Y can be BigInt/Int/long or ZFieldElExt
        """
        if isinstance(x, ZFieldElRedc):
            return self * x.inv()
        elif isinstance(x, BigInt) or isinstance(x, int) or isinstance(x, long):
            return self * ZField.inv(x)
        else:
            assert False, "Invalid type"

    def inv(self):
        """
         Compute inverse of X
        """
        return ZField.inv(self.bignum)


class ZFieldElRedc(ZFieldEl):

    def __init__(self, bignum, convert=False):
        """
         Constructor

         Parameters
         -----------
          bignum : BigInt/ZFieldElRedc/ZFieldElExt -> Initializes element
        """
        if not ZField.is_init():
            assert False, "Finite Field not initialized"

        if bignum is None:
            ZFieldEl.__init__(None)

        elif not isinstance(bignum, BigInt) and \
                not isinstance(bignum, int) and \
                not isinstance(bignum, long):
            assert False, "Incorrect Finite Field element format"
        else:
            ZFieldEl.__init__(self, bignum % ZField.get_extended_p().as_long())

    def reduce(self):
        """
         Do nothing
        """
        return ZFieldElRedc(self)

    def extend(self):
        """
         Converts Montgomery representation to default. ZFieldElExt object is returned
        """
        reduction_data = ZField.get_reduction_data()
        extended_z = ZFieldElExt((self.as_long() * reduction_data['Rp']))

        return extended_z

    # Arithmetic operators
    # * pow
    def __mul__(self, x):
        """
          X * Y (mod P) : returns ZFieldElRedc
            Y can be BigInt/Int/long or ZFieldElRedc

              NOTE : function based on https://www.nayuki.io/page/montgomery-reduction-algorithm
        """
        if isinstance(x, ZFieldElRedc):
            x_l = x.as_long()  # Montgomery multiplicatoin and reduction
        elif isinstance(x, BigInt):
            # standard multiplication
            return ZFieldElRedc((self.bignum * x.as_long()))
        elif isinstance(x, int) or isinstance(x, long):
            # standard multiplication
            return ZFieldElRedc((self.bignum * x) )
        else:
            return x * self

        mod = ZField.get_extended_p().as_long()
        reduction_data = ZField.get_reduction_data()

        product = x_l * self.as_long()
        temp = ((product & reduction_data['Rmask']) * reduction_data['Pp']) & reduction_data['Rmask']
        reduced = (product + temp * mod) >> reduction_data['Rbitlen']
        result = reduced if (reduced < mod) else (reduced - mod)

        return ZFieldElRedc(result)

    def __rmul__(self, x):
        return self * x


    def __pow__(self, x):
        """
         X ** Y (mod P) : returns ZFieldElRedc
           Y can be BigInt/Int/long or ZFieldElExt

             NOTE : function based on https://www.nayuki.io/page/montgomery-reduction-algorithm
        """
        if isinstance(x, BigInt):
            x_l = x.as_long()
            if isinstance(x, ZFieldElRedc):
                assert False, "Invalid type"
            elif x < 0:
                assert False, "Negative exponent"
        elif isinstance(x, int) or isinstance(x, long):
            x_l = x
        else:
            assert False, "Invalid type"

        tmpEl = ZFieldElRedc(self)
        z = ZFieldElRedc(ZField.get_reduction_data()['RmodP'])
        while x_l != 0:
            if x_l & 1 != 0:
                z = (tmpEl * z)
            tmpEl = tmpEl * tmpEl
            x_l >>= 1
        return z


    def __div__(self, x):
        """
          X / Y :
        """
        if isinstance(x, ZFieldElExt):
            return self * ZField.inv(x)  # standard division
        elif isinstance(x, BigInt) or isinstance(x, int) or isinstance(x, long):
            return self * x.inv()  # Montgomery division
        else:
            assert False, "Invalid type"


    def inv(self):
        """
          returns X' such that X * X' = 1 (mod p).

        """
        inv_redc = ZFieldElRedc(ZField.inv(self.bignum))
        reduction_data = ZField.get_reduction_data()

        return inv_redc * ZFieldElRedc(reduction_data['R3modP'])
