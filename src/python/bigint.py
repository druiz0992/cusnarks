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

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : BigInt
//
// Date       : 14/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//   BigInt implements functionality to operate on multi precision numbers. Python
//     provice type long with this same functionality. The purpose of the class is
//     to encapsulate required functionality so that it can be later implemented in C++/CUDA
//
//  
//   Operators:
//     Arithmetic :  +, -, *, //, %, pow 
//     Bitwise    :  <<, >>, <<=, >>=, &, |
//     Comparison :  <, <=, >, >=, ==, !=
//   show : print
//
//
//  TODO
//    - x ** y : Only works if y < POW_THR (10000). Else, raises assert
//    - x / y  : Not working
//    - Substitute assert by exception
//    
// ------------------------------------------------------------------

"""

from random import randint
import numpy as np


class BigInt(object):
    """
      BigInt class
    """
    POW_THR = 10000
    WORDS_IN_256BN = 8
    BMASK_32BIT = 0xFFFFFFFF

    def __init__(self, bignum, min_bignum=None):
        """
         Initialization

         Parameters
         -----------
          bignum  : type long, int, BigInt, string containing decimal of hex number (prececed by "0x")
          min_bignum : (Optional) same format as bignum. If different to None, created BigInt is random between
           [min_bignum and bignum]

         Notes
         -----
          if argument type different from expected, an assertion is raised

        """
        self.bignum = None
        lthr = None

        # Check min_bignum format
        # Check is min_bignum is string and convert to long format
        if min_bignum is not None:
            if isinstance(bignum, str):
                if min_bignum[:2].upper() == "0X":
                    try:
                        lthr = long(min_bignum, 16)
                    except ValueError:
                        assert False, "min_bignum string not a hexadecimal number"
                else:
                    try:
                        lthre = long(min_bignum, 10)
                    except ValueError:
                        assert False, "min_bignum string not a decimal number"

            elif isinstance(min_bignum, BigInt):
                lthr = long(min_bignum.bignum)

            elif isinstance(min_bignum, long) or isinstance(min_bignum, int):
                lthr = long(min_bignum)

            else:
                assert False, "Invalid type min_bignum"

        # Check if input is string and convert to long format
        if isinstance(bignum, str):
            if bignum[:2].upper() == "0X":
                try:
                    self.bignum = long(bignum, 16)
                except ValueError:
                    assert False, "String not a hexadecimal number"
            else:
                try:
                    self.bignum = long(bignum, 10)
                except ValueError:
                    assert False, "String not a decimal number"

        elif isinstance(bignum, BigInt):
            self.bignum = bignum.bignum

        elif isinstance(bignum, long) or isinstance(bignum, int):
            self.bignum = long(bignum)

        else:
            assert False, "Invalid type"

        if lthr is not None:
            self.bignum = randint(lthr, self.bignum)

    # Arithmetic operators
    # +, -, *, //, %, pow, +=, -=, neg
    def __add__(self, x):
        """
          X + Y
        """
        if isinstance(x, BigInt):
            return BigInt(self.bignum + x.bignum)

        elif isinstance(x, int) or isinstance(x, long):
            return BigInt(self.bignum + x)

        else:
            assert False, "Invalid type"

    def __sub__(self, x):
        """
          X - Y
        """
        if isinstance(x, BigInt):
            return BigInt(self.bignum - x.bignum)
        elif isinstance(x, int) or isinstance(x, long):
            return BigInt(self.bignum - x)
        else:
            assert False, "Invalid type"

    def __mul__(self, x):
        """
          X * Y
        """
        if isinstance(x, BigInt):
            return BigInt(self.bignum * x.bignum)
        elif isinstance(x, int) or isinstance(x, long):
            return BigInt(self.bignum * x)
        else :
            return x * self

    def __pow__(self, x):
        """
         X ^ Y
        """
        if isinstance(x, int) or isinstance(x, long) and x <= BigInt.POW_THR:
            return BigInt(self.bignum ** x)
        elif isinstance(x, BigInt) and x <= BigInt.POW_THR:
            return BigInt(self.bignum ** x.bignum)
        else:
            assert False, "Invalid type"

    def __floordiv__(self, x):
        """
         X // Y
        """
        if isinstance(x, int) or isinstance(x, long):
            return BigInt(self.bignum // x)
        elif isinstance(x, BigInt):
            return BigInt(self.bignum // x.bignum)
        else:
            assert False, "Invalid type"

    def __mod__(self, x):
        """
         X % Y
        """
        if isinstance(x, int) or isinstance(x, long):
            return BigInt(self.bignum % x)
        elif isinstance(x, BigInt):
            return BigInt(self.bignum % x.bignum)
        else:
            assert False, "Invalid type"

    def __neg__(self):
        """
         -X
        """
        return BigInt(-self.bignum)

    def __iadd__(self, x):
        """
         X += Y
        """
        if isinstance(x, BigInt):
            self.bignum += x.bignum
            return BigInt(self.bignum)
        elif isinstance(x, int) or isinstance(x, long):
            self.bignum += x
            return BigInt(self.bignum)
        else:
            assert False, "Invalid type"

    def __isub__(self, x):
        """
         X -= Y
        """
        if isinstance(x, BigInt):
            self.bignum -= x.bignum
            return BigInt(self.bignum)
        elif isinstance(x, int) or isinstance(x, long):
            self.bignum -= x
            return BigInt(self.bignum)
        else:
            assert False, "Invalid type"

    # Bitwise operators
    #  <<, >>, <<=, >>=, &, |
    def __lshift__(self, x):
        """
         X << Y
        """
        if isinstance(x, int) or isinstance(x, long):
            return BigInt(self.bignum << x)
        elif isinstance(x, BigInt):
            return BigInt(self.bignum << x.bignum)
        else:
            assert False, "Invalid type"

    def __rshift__(self, x):
        """
         X >> Y
        """
        if isinstance(x, int) or isinstance(x, long):
            return BigInt(self.bignum >> x)
        elif isinstance(x, BigInt):
            return BigInt(self.bignum >> x.bignum)
        else:
            assert False, "Invalid type"

    def __ilshift__(self, x):
        """
         X <<= Y
        """
        if isinstance(x, int) or isinstance(x, long):
            self.bignum <<= x
            return BigInt(self.bignum)
        elif isinstance(x, BigInt):
            self.bignum <<= x.bignum
            return BigInt(self.bignum)
        else:
            assert False, "Invalid type"

    def __rshift__(self, x):
        """
         X >>= Y
        """
        if isinstance(x, int) or isinstance(x, long):
            self.bignum >>= x
            return BigInt(self.bignum)
        elif isinstance(x, BigInt):
            self.bignum >>= x.bignum
            return BigInt(self.bignum)
        else:
            assert False, "Invalid type"

    def __and__(self, x):
        """
         X & Y
        """
        if isinstance(x, int) or isinstance(x, long):
            return BigInt(self.bignum & x)
        elif isinstance(x, BigInt):
            return BigInt(self.bignum & x.bignum)
        else:
            assert False, "Invalid type"

    def __or__(self, x):
        """
         X | Y
        """
        if isinstance(x, int) or isinstance(x, long):
            return BigInt(self.bignum | x)
        elif isinstance(x, BigInt):
            return BigInt(self.bignum | x.bignum)
        else:
            assert False, "Invalid type"

    # Comparison operators
    # <, <=, >, >=, ==, !=

    def __lt__(self, y):
        """
         X < Y
        """
        if isinstance(y, BigInt):
            return self.bignum < y.bignum
        elif isinstance(y, int) or isinstance(y, long):
            return self.bignum < y
        else:
            assert False, "Invalid type"

    def __le__(self, y):
        """
         X <= Y
        """
        if isinstance(y, BigInt):
            return self.bignum <= y.bignum
        elif isinstance(y, int) or isinstance(y, long):
            return self.bignum <= y
        else:
            assert False, "Invalid type"

    def __eq__(self, y):
        """
         X == Y
        """
        if isinstance(y, BigInt):
            return self.bignum == y.bignum
        elif isinstance(y, int) or isinstance(y, long):
            return self.bignum == y
        else:
            assert False, "Invalid type"

    def __ne__(self, y):
        """
         X != Y
        """
        if isinstance(y, BigInt):
            return self.bignum != y.bignum
        elif isinstance(y, int) or isinstance(y, long):
            return self.bignum != y
        else:
            assert False, "Invalid type"

    def __gt__(self, y):
        """
         X > Y
        """
        if isinstance(y, BigInt):
            return self.bignum > y.bignum
        elif isinstance(y, int) or isinstance(y, long):
            return self.bignum > y
        else:
            assert False, "Invalid type"

    def __ge__(self, y):
        """
         X >= Y
        """
        if isinstance(y, BigInt):
            return self.bignum >= y.bignum
        elif isinstance(y, int) or isinstance(y, long):
            return self.bignum >= y
        else:
            assert False, "Invalid type"

    def show(self):
        """
          print bignum
        """
        print self.bignum

    def as_long(self):
        """
           return bignum as long
        """
        return long(self.bignum)

    @classmethod
    def from_uint256(cls, bn_uint256):
        """
           convert from numpy array represenation to big int
        """
        if not isinstance(bn_uint256, np.ndarray):
            assert False, "Expected numpy array"

        elif bn_uint256.ndim != 1 or bn_uint256.shape[0] != BigInt.WORDS_IN_256BN:
            assert False, "Unexpected dimensions"

        n = np.asarray([long(x) * long(1 << 32 * i) for i, x in enumerate(bn_uint256)])
        return BigInt( np.sum(n))

    @classmethod
    def addu256(cls, x, y):
        """
           Sum 2 numbers of 256 bit
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            assert False, "Expected numpy array"

        x2 = x.reshape((-1, BigInt.WORDS_IN_256BN)).astype(np.uint64)
        y2 = y.reshape((-1, BigInt.WORDS_IN_256BN)).astype(np.uint64)
        z2 = np.zeros((x2.shape[0],x2.shape[1]+1), dtype=np.uint64)
        for i in range(BigInt.WORDS_IN_256BN):
            z2[:,i] = x2[:,i] + y2[:,i] + z2[:,i]
            z2[z2[:,i] >> 32 != 0,i+1]=1
            z2[:,i]= z2[:,i]& ((1 << 32) - 1)
        c = z2[:,-1].astype(np.uint32)
        z2 = z2[:,:-1].astype(np.uint32)
        z2.reshape(x.shape)
        return z2,c

    @classmethod
    def subu256(cls, x, y):
        """
           Sub 2 numbers of 256 bit
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            assert False, "Expected numpy array"

        x2 = x.reshape((-1, BigInt.WORDS_IN_256BN)).astype(np.uint64)
        y2 = y.reshape((-1, BigInt.WORDS_IN_256BN)).astype(np.uint64)
        z2 = np.zeros((x2.shape[0],x2.shape[1]+1), dtype=np.uint64)
        for i in range(BigInt.WORDS_IN_256BN):
            z2[:,i] = x2[:,i] - y2[:,i] + z2[:,i]
            z2[z2[:,i] >> 32 != 0,i+1]=-1
            z2[:,i]= z2[:,i]& ((1 << 32) - 1)
        z2 = z2[:,:-1].astype(np.uint32)
        c = z2[:,-1].astype(np.uint32)
        z2.reshape(x.shape)
        return z2,c
    
    @classmethod
    def addmu256(cls, x, y, p, premod=True):
       """
          Modular addition 2 numbers 256 bit
       """
       if premod:
           x_ = BigInt.modu256(x,p)
           y_ = BigInt.modu256(y,p)
       else :
           x_ = np.copy(x)
           y_ = np.copy(y)
       p_ = np.tile(p,(y.shape[0],1))

       # return submu256(x,p-y,p)
       idx    = BigInt.eq0u256(y_)
       z1, _ = BigInt.subu256(p_,y_)
       z     = BigInt.submu256(x_,z1,p)

       if any(idx):
           z[idx] = x_[idx]

       return z

    @classmethod
    def addmu256_reduce(cls, x, y, p):
       """
          Modular addition 2 numbers 256 bit
       """
       x_ = np.copy(x)
       y_ = np.copy(y)
       z = []
       do_premod = True
       while len(z) != 1:
           z = BigInt.addmu256(x_,y_,p, do_premod)
           do_premod = False
           x_ = z[0::2] 
           y_ = z[1::2] 
           if len(z)%2 == 1:
              y_ = np.concatenate((y_, np.zeros((1,BigInt.WORDS_IN_256BN), dtype=np.uint32)),axis=0)
               

       return z
    @classmethod
    def eq0u256(cls,x):
        """

        :param x: u256
        :return: x==0
        """
        x2 = x.reshape((-1, BigInt.WORDS_IN_256BN))
        return np.sum(x2,axis=1) == 0

    @classmethod
    def submu256(cls, x, y, p, premod=True):
       """
          Modular sub 2 numbers 256 bit
       """
       if premod:
           x_ = BigInt.modu256(x,p)
           y_ = BigInt.modu256(y,p)
       else :
           x_ = np.copy(x)
           y_ = np.copy(y)
       z, _ =  BigInt.subu256(x_,y_)
       z2, _ =  BigInt.subu256(np.tile(p,(y_.shape[0],1)),y_)
       z2, _ =  BigInt.addu256(z2,x_)
       idx = BigInt.ltu256(x_,y_)

       if any(idx):
          z[idx] =  z2[idx]

       return z

    @classmethod
    def modu256(cls, x, p):
       """
          Modular sub 2 numbers 256 bit
       """
       z = np.copy(x)
       x_ = np.copy(x)
       p_ = np.tile(p,(x.shape[0],1))
       while True:
          idx = BigInt.ltu256(x_,p_)
          if any(idx):
             z[idx] = x_[idx]
          if all(idx):
              break
          x_[idx==False], _ =  BigInt.subu256(x_[idx==False],p_[idx==False])

       return z

    @classmethod
    def ltu256(cls, x, y):
        """

        :param x: u256
        :param y: u256
        :return: x < y
        """
        x2 = x.reshape((-1, BigInt.WORDS_IN_256BN))
        y2 = y.reshape((-1, BigInt.WORDS_IN_256BN))
        z = np.zeros(x2.shape[0], dtype=np.uint32)
        for i in range(BigInt.WORDS_IN_256BN-1,0,-1):
            z[np.bitwise_or(x2[:,i] > y2[:,i], z==0)] = 2
            z[np.bitwise_or(x2[:,i] < y2[:,i], z==0)] = 1
            if not any(z==0):
                break

        return z == 1
    def as_uint256(self):
       """ 
         return big int as numpy array of uint32
       """
       bn = self.bignum
       return np.asarray([bn & BigInt.BMASK_32BIT,
                             (bn >> 32 ) & BigInt.BMASK_32BIT,
                             (bn >> 64 ) & BigInt.BMASK_32BIT,
                             (bn >> 96 ) & BigInt.BMASK_32BIT,
                             (bn >> 128) & BigInt.BMASK_32BIT,
                             (bn >> 160) & BigInt.BMASK_32BIT,
                             (bn >> 192) & BigInt.BMASK_32BIT,
                             (bn >> 224) & BigInt.BMASK_32BIT], dtype=np.uint32)

       


