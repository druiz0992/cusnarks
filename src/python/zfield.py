"""
// ------------------------------------------------------------------
// (c) Copyright 2019, Toboso Networks                               
// All Rights Reserved                                               
//                                                                   
//
// This program is the proprietary software of Toboso Networks and/or
// its licensors, and may only be used, duplicated, modified or distributed
// pursuant to the terms and conditions of a separate, written license
// agreement executed between you and Toboso Networks (an "Authorized License").
// Except as set forth in an Authorized License, Toboso Networks grants no license
// (express or implied), right to use, or waiver of any kind with respect to
// the Software, and Toboso Networks expressly reserves all rights in and to the
// Software and all intellectual property rights therein.  IF YOU HAVE NO
// AUTHORIZED LICENSE, THEN YOU HAVE NO RIGHT TO USE THIS SOFTWARE IN ANY
// WAY, AND SHOULD IMMEDIATELY NOTIFY TOBOSO NETWORKS AND DISCONTINUE ALL USE OF
// THE SOFTWARE.
//
// ..................................................................
// ... 000 ......... 000 ........ t ....... b .......................
// ..... 000 ..... 000 ........ ttttt ooooo bbbbb ooooo sssss ooooo .
// ....... 000 . 000 ............ t . o . o b . b o . o s ... o . o .
// .......... 000||| ............ t . o . o b . b o . o sssss o . o .
// ....... 000|||OOO|| .......... t . o . o b . b o . o ... s o . o .
// ..... 000|||||||OOO|| ........ ttt ooooo bbbbb ooooo sssss ooooo .
// ... 000|||||||||||OOO| .............. t ..........................
// ...... ||||||||||||||| ..... nnn  eee ttt w  w oooo rrr k k ssss .
// ...... ||||||||||||||| ..... n  n e . t . www  o  o r . kk . ss ..
// ...... ||||||||||||||| ..... n  n eee ttt  w . oooo r . k k ssss .
// ..... ||||||||||||||||| ..........................................
// ..... ||||||||||||||||| ..........................................
// ..... ||||||||||||||||| ..........................................
// ..................................................................
//  Toboso Networks ........................... Strictly Confidential
// ..................................................................
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
//          @classmethod get(self) : Returns prime
//          @classmethod set(self,p) : Changes prime - Not recommended to change prime. ZFieldEl
//             initialied with previous prime will stop making sense.
//          @classmethod is_init(self) : True if ZField is initialized
//          @classmethod is_redc(self) : True if ZField prime has been reduced (Montgomery reduction)
//          @classmethod show(self) : Print ZField prime
//          @classmethod reduce(self) : Reduce ZField prime (Montgomery)
//          @classmethod extend(self) : Extend ZField prime
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

class ZField:
  init_prime = False
  redc_prime = False
  prime = 0
  prime_ext = prime

  # Montgomery data
  reducerbits = 0
  reducer = 0
  mask = 0
  reciprocal = 0
  factor = 0
  convertedone = 0


  def __init__(self,q):
      self.set(q)

  @classmethod
  def get(self):
    return ZField.prime.bignum

  @classmethod
  def set(self,q):
    ZField.prime = BigInt(q)
    ZField.prime_ext = ZField.prime
    ZField.init_prime = True
    ZField.redc_prime = False

  @classmethod
  def is_init(self):
      return ZField.init_prime

  @classmethod
  def is_redc(self):
      return ZField.redc_prime

  @classmethod
  def show(self):
      ZField.prime.show()

  @classmethod
  def reduce(self):
      if not ZField.redc_prime:
          bitlen = int(math.ceil(math.log(ZField.get(),2)))
          ZField.prime = BigInt(1 << bitlen | 1) # force it to be odd
          ZField.reducerbits = (ZField.get().bit_length() // 8 +1 )* 8 # Multiple of 8
          ZField.reducer = 1 << ZField.reducerbits
          ZField.mask = ZField.reducer - 1
          ZField.reciprocal = ZFieldEl._inv(ZField.reducer % ZField.get())
          ZField.factor = (ZField.reducer * ZField.reciprocal - 1) // ZField.get()
          ZField.convertedone = ZField.reducer % ZField.get()

      ZField.redc_prime = True

  @classmethod
  def extend(self):
      if ZField.redc_prime:
         ZField.prime = ZField.prime_ext

      ZField.redc_prime = False


class ZFieldEl(BigInt): 
   def __init__(self, bignum):
     """
      Initialization

      Parameters
      -----------
       bignum 
     """
     if ZField.is_init():
         self.bignum = BigInt(bignum) % ZField.get()
         if ZField.is_redc():
           self.reduce()
           self.is_redc = True
         else:
             self.is_redc = False
     else :
       assert True, "Prime not initialized"

   def reduce(self):
       if not self.is_redc:
           self.bignum = (self.bignum << ZField.reducerbits) % ZField.get()
           self.is_redc = True

   def extend(self):
       if self.is_redc:
           self.bignum = (self.bignum * ZField.reciprocal) % ZField.get()
           self.is_redc = False

   # Arithmetic operators
   # +, -, *, /, %, pow
   def __add__(self,x):
    """
    """
    if isinstance(x,BigInt):
      return (self.bignum + x.bignum) % ZField.get()
    elif isinstance(x,int) or isinstance(x,long):
      return (self.bignum + x) % ZField.get()
    else :
      assert True,"Invalid type"

   def __sub__(self,x):
    if isinstance(x,BigInt):
     return (self.bignum - x.bignum) % ZField.get()
    elif isinstance(x,int) or isinstance(x,long):
      return (self.bignum - x) % Zfield.get()

   def __mul__(self,x):
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
     if isinstance(x,int) or isinstance(x,long):
       return (self.bignum // x) % ZField.get()
     elif isinstance(x,BigInt):
       return (self.bignum // x.bignum) % ZField.get()
     else :
       assert True,"Invalid type"

   def __truediv__ (self,x):
     if isinstance(x,int) or isinstance(x,long):
       return (self.bignum // x) % ZField.get()
     elif isinstance(x,BigInt):
       return (self.bignum // x.bignum) % ZField.get()
     else :
       assert True,"Invalid type"

   def __mod__ (self,x):
     if isinstance(x,int) or isinstance(x,long):
       return (self.bignum % x) % ZField.get()
     elif isinstance(x,BigInt):
       return (self.bignum % x.bignum) % ZField.get()
     else :
       assert True,"Invalid type"

   def __iadd__(self,x):
     if isinstance(x,BigInt):
       self.bignum = (self.bignum + x.bignum) % ZField.get()
       return self
     elif isinstance(x,int) or isinstance(x,long):
       self.bignum = (self.bignum + x) % ZField.get()
       return self
     else :
       assert True,"Invalid type"

   def __isub__(self,x):
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

     mod = ZField.get()
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
     if isinstance(x,BigInt):
       return self.bignum < x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum < x % ZField.get()
     else :
       assert True,"Invalid type"

   def __le__(self,x):
     if isinstance(x,BigInt):
       return self.bignum <= x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum <= x % ZField.get()
     else :
       assert True,"Invalid type"

   def __eq__(self,x):
     if isinstance(x,BigInt):
       return self.bignum == x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum == x % ZField.get()
     else :
       assert True,"Invalid type"

   def __ne__(self,x):
     if isinstance(x,BigInt):
       return self.bignum != x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum != x % ZField.get()
     else :
       assert True,"Invalid type"

   def __gt__(self,x):
     if isinstance(x,BigInt):
       return self.bignum > x.bignum % ZField.get()
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum > x % ZField.get()
     else :
       assert True,"Invalid type"

   def __ge__(self,x):
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
       temp = ((product & ZField.mask) * ZField.factor) & ZField.mask
       reduced = (product + temp * mod) >> ZField.reducerbits
       result = reduced if (reduced < mod) else (reduced - mod)
       assert 0 <= result < mod
       return result

   @staticmethod
   def _pow_redc(x,y):
       assert 0 <= x < ZField.get()
       if y < 0:
           raise ValueError("Negative exponent")
       z = ZField.convertedone
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
