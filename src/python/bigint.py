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


class BigInt:
   """
     BigInt class
   """
   POW_THR = 10000   
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
         if isinstance(bignum,str):
             if min_bignum[:2].upper() == "0X":
               try:
                 lthr = long(min_bignum,16)
               except ValueError:
                 assert True,"min_bignum string not a hexadecimal number"
             else:
               try:
                 lthre = long(min_bignum,10)
               except ValueError:
                 assert True,"min_bignum string not a decimal number"
            
         elif isinstance(min_bignum,BigInt):
            lthr = long(min_bignum.bignum)
      
         elif isinstance(min_bignum,long) or isinstance(min_bignum,int):
            lthr = long(min_bignum)
    
         else :
           assert True,"Invalid type min_bignum"

     # Check if input is string and convert to long format
     if isinstance(bignum,str):
         if bignum[:2].upper() == "0X":
           try:
             self.bignum = long(bignum,16)
           except ValueError:
             assert True,"String not a hexadecimal number"
         else:
           try:
             self.bignum = long(bignum,10)
           except ValueError:
             assert True,"String not a decimal number"
     
     elif isinstance(bignum,BigInt):
        self.bignum = bignum.bignum
  
     elif isinstance(bignum,long) or isinstance(bignum,int):
        self.bignum = long(bignum)
    
     else :
       assert True,"Invalid type"

     if lthr is not None:
        self.bignum = randint(lthr,self.bignum)

   # Arithmetic operators
   # +, -, *, //, %, pow, +=, -=, neg
   def __add__(self,x):
    """
      X + Y
    """
    if isinstance(x,BigInt):
      return BigInt(self.bignum + x.bignum)

    elif isinstance(x,int) or isinstance(x,long):
      return BigInt(self.bignum + x)

    else :
      assert True,"Invalid type"

   def __sub__(self,x):
    """
      X - Y
    """
    if isinstance(x,BigInt):
     return BigInt(self.bignum - x.bignum)
    elif isinstance(x,int) or isinstance(x,long):
      return BigInt(self.bignum - x)

   def __mul__(self,x):
    """
      X * Y
    """
    if isinstance(x,BigInt):
      return BigInt(self.bignum * x.bignum)
    elif isinstance(x,int) or isinstance(x,long):
      return BigInt(self.bignum * x)

   def __pow__ (self,x): 
     """
      X ^ Y
     """
     if isinstance(x,int) or isinstance(x,long) and <= BigInt.POW_THR:
       return BigInt(self.bignum ** x)
     elif isinstance(x,BigInt) and x <= BigInt.POW_THR :
       return BigInt(self.bignum ** x.bignum)
     else :
       assert True,"Invalid type"
   
   def __floordiv__ (self,x):
     """
      X // Y
     """
     if isinstance(x,int) or isinstance(x,long):
       return BigInt(self.bignum // x)
     elif isinstance(x,BigInt):
       return BigInt(self.bignum // x.bignum)
     else :
       assert True,"Invalid type"

   def __truediv__ (self,x):
     """
       X // Y
       x / y  : Not working
     """
     if isinstance(x,int) or isinstance(x,long):
       return BigInt(self.bignum // x)
     elif isinstance(x,BigInt):
       return BigInt(self.bignum // x.bignum)
     else :
       assert True,"Invalid type"

   def __mod__ (self,x):
     """
      X % Y
     """
     if isinstance(x,int) or isinstance(x,long):
       return BigInt(self.bignum % x)
     elif isinstance(x,BigInt):
       return BigInt(self.bignum % x.bignum)
     else :
       assert True,"Invalid type"

   def __neg__ (self):
     """
      -X 
     """
     return BigInt(-self.bignum)

   def __iadd__(self,x):
     """
      X += Y
     """
     if isinstance(x,BigInt):
       self.bignum += x.bignum
       return BigInt(self.bignum)
     elif isinstance(x,int) or isinstance(x,long):
       self.bignum += x
       return BigInt(self.bignum)
     else :
       assert True,"Invalid type"

   def __isub__(self,x):
     """
      X -= Y
     """
     if isinstance(x,BigInt):
       self.bignum -= x.bignum
       return BigInt(self.bignum)
     elif isinstance(x,int) or isinstance(x,long):
       self.bignum -= x
       return BigInt(self.bignum)
     else :
       assert True,"Invalid type"

   # Bitwise operators
   #  <<, >>, <<=, >>=, &, |
 
   def __lshift__(self,x):
     """
      X << Y
     """
     if isinstance(x,int) or isinstance(x,long):
       return BigInt(self.bignum << x)
     elif isinstance(x,BigInt):
       return BigInt(self.bignum << x.bignum)
     else :
       assert True,"Invalid type"
  
   def __rshift__(self,x):
     """
      X >> Y
     """
     if isinstance(x,int) or isinstance(x,long):
       return BigInt(self.bignum >> x)
     elif isinstance(x,BigInt):
       return BigInt(self.bignum >> x.bignum)
     else :
       assert True,"Invalid type"

   def __ilshift__(self,x):
     """
      X <<= Y
     """
     if isinstance(x,int) or isinstance(x,long):
       self.bignum <<= x
       return BigInt(self.bignum)
     elif isinstance(x,BigInt):
       self.bignum <<= x.bignum
       return BigInt(self.bignum)
     else :
       assert True,"Invalid type"
  
   def __rshift__(self,x):
     """
      X >>= Y
     """
     if isinstance(x,int) or isinstance(x,long):
       self.bignum >>= x
       return BigInt(self.bignum)
     elif isinstance(x,BigInt):
       self.bignum >>= x.bignum
       return BigInt(self.bignum)
     else :
       assert True,"Invalid type"


   def __and__(self,x):
     """
      X & Y
     """
     if isinstance(x,int) or isinstance(x,long):
       return BigInt(self.bignum & x)
     elif isinstance(x,BigInt):
       return BigInt(self.bignum & x.bignum)
     else :
       assert True,"Invalid type"
   
   def __or__(self,x):
     """
      X | Y
     """
     if isinstance(x,int) or isinstance(x,long):
       return BigInt(self.bignum | x)
     elif isinstance(x,BigInt):
       return BigInt(self.bignum | x.bignum)
     else :
       assert True,"Invalid type"
  

   # Comparison operators
   # <, <=, >, >=, ==, !=

   def __lt__(self,y):
     """
      X < Y
     """
     if isinstance(y,BigInt):
       return self.bignum < y.bignum
     elif isinstance(y,int) or isinstance(y,long):
       return self.bignum < y
     else :
       assert True,"Invalid type"

   def __le__(self,y):
     """
      X <= Y
     """
     if isinstance(y,BigInt):
       return self.bignum <= y.bignum
     elif isinstance(y,int) or isinstance(y,long):
       return self.bignum <= y
     else :
       assert True,"Invalid type"

   def __eq__(self,y):
     """
      X == Y
     """
     if isinstance(y,BigInt):
       return self.bignum == y.bignum
     elif isinstance(y,int) or isinstance(y,long):
       return self.bignum == y
     else :
       assert True,"Invalid type"

   def __ne__(self,y):
     """
      X != Y
     """
     if isinstance(y,BigInt):
       return self.bignum != y.bignum
     elif isinstance(y,int) or isinstance(y,long):
       return self.bignum != y
     else :
       assert True,"Invalid type"

   def __gt__(self,y):
     """
      X > Y
     """
     if isinstance(y,BigInt):
       return self.bignum > y.bignum
     elif isinstance(y,int) or isinstance(y,long):
       return self.bignum > y
     else :
       assert True,"Invalid type"

   def __ge__(self,y):
     """
      X >= Y
     """
     if isinstance(y,BigInt):
       return self.bignum >= y.bignum
     elif isinstance(y,int) or isinstance(y,long):
       return self.bignum >= y
     else :
       assert True,"Invalid type"

   def show (self):
       """
         print bignum
       """
       print self.bignum

   def as_long(self):
      """
         return bignum as long
      """
      return long(self.bignum)

