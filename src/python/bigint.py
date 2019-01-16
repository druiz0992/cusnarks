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
// File name  : BigInt
//
// Date       : 14/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//   BigInt implements functionality to operate on multi precission numbers
//  
//   Operators:
//     Arithmetic :  +, -, *, /, %, pow 
//     Logical    : //  <<, >>, &, |
//     Comparison : # <, <=, >, >=, ==, !=
//   random
//   show
//
//  TODO
//    - x ** y : Not working when y is bignum
//    - x / y  : Not working
//    - 
// ------------------------------------------------------------------

"""

from random import randint


class BigInt:
   """
     BigInt class
   """
   def __init__(self, bignum):
     """
      Initialization

      Parameters
      -----------
       bignum 
     """
     self.bignum = None

     if isinstance(bignum,str):
       try:
         self.bignum = long(bignum,16)
       except ValueError:
         assert True,"String not a hexadecimal number"
        
     
     elif isinstance(bignum,BigInt):
        self.bignum = bignum.bignum
  
     elif isinstance(bignum,long) or isinstance(bignum,int):
        self.bignum = bignum

     else :
       assert True,"Invalid type"

   # Arithmetic operators
   # +, -, *, /, %, pow
   def __add__(self,x):
    """
    """
    if isinstance(x,BigInt):
      return self.bignum + x.bignum
    elif isinstance(x,int) or isinstance(x,long):
      return self.bignum + x
    else :
      assert True,"Invalid type"

   def __sub__(self,x):
    if isinstance(x,BigInt):
     return self.bignum - x.bignum
    elif isinstance(x,int) or isinstance(x,long):
      return self.bignum - x

   def __mul__(self,x):
    if isinstance(x,BigInt):
      return self.bignum + x.bignum
    elif isinstance(x,int) or isinstance(x,long):
      return self.bignum * x

   def __pow__ (self,x): 
     if isinstance(x,int) or isinstance(x,long):
       return self.bignum ** x
     elif isinstance(x,BigInt):
       assert True,"Invalid type"
     else :
       assert True,"Invalid type"
   
   def __floordiv__ (self,x):
     if isinstance(x,int) or isinstance(x,long):
       return self.bignum // x
     elif isinstance(x,BigInt):
       return self.bignum // x.bignum
     else :
       assert True,"Invalid type"

   def __truediv__ (self,x):
     if isinstance(x,int) or isinstance(x,long):
       return self.bignum // x
     elif isinstance(x,BigInt):
       return self.bignum // x.bignum
     else :
       assert True,"Invalid type"

   def __mod__ (self,x):
     if isinstance(x,int) or isinstance(x,long):
       return self.bignum % x
     elif isinstance(x,BigInt):
       return self.bignum % x.bignum
     else :
       assert True,"Invalid type"

   def __iadd__(self,x):
     if isinstance(x,BigInt):
       self.bignum += x.bignum
       return BigInt(self.bignum)
     elif isinstance(x,int) or isinstance(x,long):
       self.bignum += x
       return BigInt(self)
     else :
       assert True,"Invalid type"

   def __isub__(self,x):
     if isinstance(x,BigInt):
       self.bignum -= x.bignum
       return BigInt(self.bignum)
     elif isinstance(x,int) or isinstance(x,long):
       self.bignum -= x
       return BigInt(self)
     else :
       assert True,"Invalid type"

   # Logical operators
   #  <<, >>, <<=, >>=, &, |
 
   def __lshift__(self,x):
     if isinstance(x,int) or isinstance(x,long):
       return self.bignum << x
     elif isinstance(x,BigInt):
       return self.bignum << x.bignum
     else :
       assert True,"Invalid type"
  
   def __rshift__(self,x):
     if isinstance(x,int) or isinstance(x,long):
       return self.bignum >> x
     elif isinstance(x,BigInt):
       return self.bignum >> x.bignum
     else :
       assert True,"Invalid type"

   def __and__(self,x):
     if isinstance(x,int) or isinstance(x,long):
       return self.bignum & x
     elif isinstance(x,BigInt):
       return self.bignum & x.bignum
     else :
       assert True,"Invalid type"
   
   def __or__(self,x):
     if isinstance(x,int) or isinstance(x,long):
       return self.bignum | x
     elif isinstance(x,BigInt):
       return self.bignum | x.bignum
     else :
       assert True,"Invalid type"
  

   # Comparison operators
   # <, <=, >, >=, ==, !=

   def __lt__(self,x):
     if isinstance(x,BigInt):
       return self.bignum < x.bignum
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum < x
     else :
       assert True,"Invalid type"

   def __le__(self,x):
     if isinstance(x,BigInt):
       return self.bignum <= x.bignum
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum <= x
     else :
       assert True,"Invalid type"

   def __eq__(self,x):
     if isinstance(x,BigInt):
       return self.bignum == x.bignum
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum == x
     else :
       assert True,"Invalid type"

   def __ne__(self,x):
     if isinstance(x,BigInt):
       return self.bignum != x.bignum
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum != x
     else :
       assert True,"Invalid type"

   def __gt__(self,x):
     if isinstance(x,BigInt):
       return self.bignum > x.bignum
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum > x
     else :
       assert True,"Invalid type"

   def __ge__(self,x):
     if isinstance(x,BigInt):
       return self.bignum >= x.bignum
     elif isinstance(x,int) or isinstance(x,long):
       return self.bignum >= x
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

     return randint(minn,maxn)

   def show (self):
       print self.bignum

   def get(self):
      return self.bignum

if __name__ == '__main__':
    x = BigInt("0x12345a12345678902345")
    y = BigInt("0x12345a145678902345")
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
