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
