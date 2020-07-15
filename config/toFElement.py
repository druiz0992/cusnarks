#!/usr/bin/python3

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
// File name  : process_ff.py
//
// Date       : 12/05/2020
//
// ------------------------------------------------------------------
//
// Description:
//  Process FF and convert them to appropriate format
//
// ------------------------------------------------------------------

"""
import sys
import os
import ast
 
sys.path.append(os.path.abspath(os.path.dirname('../src/python/')))

from zfield import *
from z2field_element import *


class FF(object):

   def __init__(self, fr, fp, G1x, G1y, G2x1, G2y1, G2x2, G2y2, factors_fr):
      self.f = open("test.dat","w")
      ZField.init_prime = False
    
      fd = ast.literal_eval(factors_fr)
      zp       = ZField(fr, fd['factor_data'])
      self.FR   = zp.get_extended_p()
      r_data   = zp.redc_data[0]
      self.FR_NWORDS = int((r_data['Rbitlen'] + 31) / 32)
      self.FR_NP      = BigInt(r_data['Pp'])
      self.FR_R2   =  ZFieldElExt(BigInt( (r_data['R']*r_data['R']) % zp.get_extended_p().as_long()))
      self.FR_R2_rdc      = self.FR_R2.reduce()
      self.FR_One_rdc          = ZFieldElExt(1).reduce()
      self.FR_IScaler = []
      self.FR_IScalerMont = []
      self.nFrRoots = fd['factor_data']['exponents'][0]
      for i in range(self.nFrRoots+1):
         self.FR_IScaler.append(ZFieldElExt(1<<i).inv())
         self.FR_IScalerMont.append(self.FR_IScaler[i].reduce())

      pr = zp.find_primitive_root()
      self.FR_roots_rdc = []
      for i in range(self.nFrRoots+1):
        r = pow(pr.as_long(), 1<<i, self.FR.as_long())
        self.FR_roots_rdc.append(ZFieldElExt(r).reduce())
     
      self.FR_roots_rdc = self.FR_roots_rdc[::-1]

      self.printFR()

      ZField.init_prime = False
      zp       = ZField(fp)
      self.FP   = zp.get_extended_p()
      r_data   = zp.redc_data[0]
      self.FP_NWORDS = int((r_data['Rbitlen'] + 31) / 32)
      self.FP_NP      = BigInt(r_data['Pp'])
      self.FP_R2   =  ZFieldElExt(BigInt( (r_data['R']*r_data['R']) % zp.get_extended_p().as_long()))
      self.FP_R2_rdc      = self.FP_R2.reduce()
      self.FP_One_rdc          = ZFieldElExt(1).reduce()

      self.G1x  = ZFieldElExt(BigInt(G1x))
      self.G1y  = ZFieldElExt(BigInt(G1y))
      # only valid for y^2 = x^3 + b curves
      self.b = (self.G1y * self.G1y  - self.G1x * self.G1x * self.G1x)

      self.G2x1  = ZFieldElExt(BigInt(G2x1))
      self.G2x2  = ZFieldElExt(BigInt(G2x2))
      self.G2y1  = ZFieldElExt(BigInt(G2y1))
      self.G2y2  = ZFieldElExt(BigInt(G2y2))

      self.G2x  = Z2FieldEl([self.G2x1, self.G2x2])
      self.G2y  = Z2FieldEl([self.G2y1, self.G2y2])
      # only valid for y^2 = x^3 + b curves
      self.b2 = (self.G2y * self.G2y  - self.G2x * self.G2x * self.G2x)
      #print(self.b2.P[0].as_long())
      #print(self.b2.P[1].as_long())

      self.printFP()
      self.f.close()

   
   def printFR(self):
      s1 = printW32(self.FR.as_uint256(NW=self.FR_NWORDS))
      self.toFile(s1,"<FR>")
      s1 = printW32(self.FR_NP.as_uint256(NW=self.FR_NWORDS))
      self.toFile(s1,"<FR_NP>")
      s1 = printW32(self.FR_R2.as_uint256(NW=self.FR_NWORDS))
      self.toFile(s1,"<FR_R2>")
      s1 = printW32(self.FR_R2_rdc.as_uint256(NW=self.FR_NWORDS))
      self.toFile(s1,"<FR_R2_rdc>")
      s1 = printW32(self.FR_One_rdc.as_uint256(NW=self.FR_NWORDS))
      self.toFile(s1,"<FR_One_rdc>")
      s1 += ", "+printW32(np.zeros(self.FR_NWORDS, dtype=np.uint32))
      self.toFile(s1,"<FR_One2_rdc>")
      s1 = ""
      for i in range(self.nFrRoots+1):
        s1 += printW32(self.FR_IScaler[i].as_uint256(NW=self.FR_NWORDS))
        if i < self.nFrRoots:
          s1 += ", "
      self.toFile(s1,"<FR_IScaler>")

      s1 = ""
      for i in range(self.nFrRoots+1):
        s1 += printW32(self.FR_IScalerMont[i].as_uint256(NW=self.FR_NWORDS))
        if i < self.nFrRoots :
          s1 += ", "
      self.toFile(s1,"<FR_IScalerMont>")

      s1 = ""
      for i in range(self.nFrRoots+1):
        s1 += printW32(self.FR_roots_rdc[i].as_uint256(NW=self.FR_NWORDS))
        if i < self.nFrRoots:
          s1 += ", "
      self.toFile(s1,"<FR_FrRoots_rdc>")

      s1 = printW32(ZFieldElExt(BigInt(1)).as_uint256(NW=self.FR_NWORDS))
      self.toFile(s1,"<FR_One>")
      s1 = printW32(ZFieldElExt(BigInt(0)).as_uint256(NW=self.FR_NWORDS))
      self.toFile(s1,"<FR_Zero>")


   def printFP(self):
      s1 = printW32(self.FP.as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP>")
      s1 = printW32(self.FP_NP.as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_NP>")
      s1 = printW32(self.FP_R2.as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_R2>")
      s1 = printW32(self.FP_R2_rdc.as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_R2_rdc>")
      s1 = printW32(self.FP_One_rdc.as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_One_rdc>")
      s1 += ", "+printW32(np.zeros(self.FP_NWORDS, dtype=np.uint32))
      self.toFile(s1,"<FP_One2_rdc>")
      s2 = printW32(np.zeros(self.FP_NWORDS, dtype=np.uint32))
      s2 += ", "+s1
      self.toFile(s2,"<FP_G1_inf_rdc>")
      s1 = printW32(np.zeros(self.FP_NWORDS, dtype=np.uint32))
      s2 = s1+", "+s2+", "+s1+", "+s1

      self.toFile(s2,"<FP_G2_inf_rdc>")
     
      s1 = printW32(self.G1x.reduce().as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_G1x_rdc>")
      s1 = printW32(self.G1y.reduce().as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_G1y_rdc>")
      s1 = printW32(self.b.reduce().as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_B_rdc>")

      s1 = printW32(self.G2x1.reduce().as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_G2x1_rdc>")
      s1 = printW32(self.G2x2.reduce().as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_G2x2_rdc>")
      s1 = printW32(self.G2y1.reduce().as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_G2y1_rdc>")
      s1 = printW32(self.G2y2.reduce().as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_G2y2_rdc>")
      s1 = printW32(self.b2.P[0].reduce().as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_B2_1_rdc>")
      s1 = printW32(self.b2.P[1].reduce().as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_B2_2_rdc>")
      s1 = printW32(ZFieldElExt(1).as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_One>")
      s1 = printW32(ZFieldElExt(0).as_uint256(NW=self.FP_NWORDS))
      self.toFile(s1,"<FP_Zero>")

   def toFile(self, s, lable):
      data = lable +" " + s + "\n"
      self.f.write(data)


def printW32(data):
   s=""
   for idx, x in enumerate(data):
     s = s + str(x) 
     if idx < len(data) -1:
        s = s + ', '
   return s

if __name__ == '__main__':
    if len(sys.argv) == 10:
       fr = sys.argv[1]
       fp = sys.argv[2]
       G1x = sys.argv[3]
       G1y = sys.argv[4]
       G2x1 = sys.argv[5]
       G2x2 = sys.argv[6]
       G2y1 = sys.argv[7]
       G2y2 = sys.argv[8]
       factors = sys.argv[9]

       a = FF(fr, fp, G1x, G1y, G2x1, G2x2, G2y1, G2y2, factors)

    else :
      print("Usage : toElement.py FR, FP, G1X, G1Y, G2X1, G2X2, G2Y1, G2Y2, Factors")
    
