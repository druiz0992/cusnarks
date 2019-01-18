
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

#  NOTES:
//
//
// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : zutils
//
// Date       : 18/01/2019
//
// ------------------------------------------------------------------
//
// Description:
//    Implementation of different utility functions and constants    
//
//          @staticmethod prime_factors(n) : Returns dictionary with prime factors and exponents of 
//                    factorization of number n. If number n is very large, function may take a long time
//   TODO
//    
// ------------------------------------------------------------------

"""
import math
import numpy as np
from random import randint

class ZUtils(object):

    @staticmethod
    def find_primes(start,end, cnt=0):
      # Initialize a list
      primes = []
      for possiblePrime in xrange(start,end):
          if all((possiblePrime % i != 0) for i in range(2, int(math.floor(math.sqrt(possiblePrime))) + 1)):
              primes.append(possiblePrime)
          if len(primes) >= cnt and cnt != 0:
              break
  
      return primes

    @staticmethod
    def prime_factors(n):
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





