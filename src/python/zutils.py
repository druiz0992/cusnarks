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
// NOTES:

# 
# Factorization and finding primes code based in Project Nayuki
# 
# Copyright (c) 2018 Project Nayuki
# All rights reserved. Contact Nayuki for licensing.
# https://www.nayuki.io/page/number-theoretic-transform-integer-dft


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
from random import randint
from past.builtins import xrange

import numpy as np


class ZUtils(object):
    PREDEFINED_CURVES = ['P1009', 'Secp112r1', 'Secp128r1', 'Secp160k1', 'Secp224k1', 'Secp256k1', 'BN128']

    # y^2 = x^3 + a*x + b
    CURVE_DATA = {
        'P1009':
            {'curve': 'P1009',
             'prime': 1009,
             'factor_data': {'factors': [2, 3, 7],
                             'exponents': [4, 2, 1]}
             },
        'Secp112r1':
            {'curve': 'Sepc112r1',
             'prime': 4451685225093714772084598273548427,
             'curve_params': {'a': 4451685225093714772084598273548424, 'b': 2061118396808653202902996166388514,
                              'Gx': 188281465057972534892223778713752, 'Gy': 3419875491033170827167861896082688},
             'factor_data': {'factors': [2, 3, 7, 1453958119802281, 5207091687747401],
                             'exponents': [2, 1, 2, 1, 1]}
             },
        'Secp128r1':
            {'curve': 'Sepc128r1',
             'prime': 340282366762482138434845932244680310783,
             'curve_params': {'a': '1329227995165945853261116922830782463',
                              'b': 1206995559461897101683291410318290526,
                              'Gx': 29408993404948928992877151431649155974,
                              'Gy': 275621562871047521857442314737465260675},
             'factor_data': {'factors': [2, 2147483647],
                             'exponents': [97, 1]}
             },
        'Secp160k1':
            {'curve': 'Sepc160k1',
             'prime': 1461501637330902918203684832716283019651637554291,
             'curve_params': {'a': 0, 'b': 7,
                              'Gx': 338530205676502674729549372677647997389429898939,
                              'Gy': 842365456698940303598009444920994870805149798382},
             'factor_data': {'factors': [2, 37, 44481592398149, 222002193056774815430442568663621],
                             'exponents': [2, 1, 1, 1]}
             },
        'Secp224k1':
            {'curve': 'Sep224k1',
             'prime': 26959946667150639794667015087019630673637144422540572481099315275117,
             'curve_params': {'a': 0, 'b': 5,
                              'Gx': 16983810465656793445178183341822322175883642221536626637512293983324,
                              'Gy': 13272896753306862154536785447615077600479862871316829862783613755813},
             'factor_data': {'factors': [2, 50238476144222203, 268319709675859997416334102104367237320252177313653],
                             'exponents': [1, 2, 2]}
             },
        'Sepc256k1':
            {'curve': 'Sepc256k1',
             'prime': 115792089237316195423570985008687907853269984665640564039457584007908834671663,
             'curve_params': {'a': 0, 'b': 7,
                              'Gx': 55066263022277343669578718895168534326250603453777594175500187360389116729240,
                              'Gy': 32670510020758816978083085130507043184471273380659243275938904335757337482424},
             'factor_data': {
                 'factors': [2, 7322137, 45422601869677, 21759506893163426790183529804034058295931507131047955271],
                 'exponents': [4, 1, 1, 1]}
             },
        'BN128': {
            'curve': 'BN128',
            'prime':21888242871839275222246405745257275088696311157297823662689037894645226208583,
            'prime_r': 21888242871839275222246405745257275088548364400416034343698204186575808495617,
            'curve_params': {'a': 0, 'b': 3,
                             'Gx': 1,
                             'Gy': 2},
            #twisted curve ofver FQ**2, b = FQ2([3,0])/FQ2([9,1])
            'curve_params_g2': {'ax1': 0,
                                'ax2': 0,
                                #'bx1' : 16308358912334916840975062027523648053282252831902403662509936306329156108382,
                                #'bx2' : 12809462434952274491643852919292665322273875818504683666956364026004085567101,
                                'bx1': 19485874751759354771024239261021720505790618469301721065564631296452457478373,
                                'bx2': 266929791119991161246907387137283842545076965332900288569378510910307636690,
                                'Gx1': 10857046999023057135944570762232829481370756359578518086990519993285655852781,
                                'Gx2': 11559732032986387107991004021392285783925812861821192530917403151452391805634,
                                'Gy1': 8495653923123431417604973247489272438418190587263600148770280649306958101930,
                                'Gy2': 4082367875863433681332203403145435568316851327593401208105741076214120093531
                                },
            'factor_data': {'factors': [2, 3, 13, 29, 983, 11003, 237073, 405928799, 1670836401704629,
                                        13818364434197438864469338081],
                            'exponents': [28, 2, 1, 1, 1, 1, 1, 1, 1, 1]}
        },
    }

    AFFINE     = 0
    PROJECTIVE = 1
    JACOBIAN = 2

    DEFAULT_IN_REP_FORMAT = AFFINE

    FEXT = 0
    FRDC = 1
    DEFAULT_IN_PFORMAT = FEXT

    NROOTS = 8192

    @classmethod
    def get_NROOTS(cls):
       return ZUtils.NROOTS

    @classmethod
    def set_NROOTS(cls, nroots):
       ZUtils.NROOTS = nroots
  
    @classmethod
    def get_default_in_rep_format(cls):
        return ZUtils.DEFAULT_IN_REP_FORMAT

    @classmethod
    def set_default_in_rep_format(cls, fmt):
        if fmt == ZUtils.PROJECTIVE or fmt == ZUtils.AFFINE or fmt == ZUtils.JACOBIAN:
            ZUtils.DEFAULT_IN_REP_FORMAT = fmt
      

    @classmethod
    def get_default_in_p_format(cls):
        return ZUtils.DEFAULT_IN_PFORMAT

    @classmethod
    def set_default_in_p_format(cls, fmt):
        if fmt == ZUtils.FEXT or fmt == ZUtils.FRDC:
            ZUtils.DEFAULT_IN_PFORMAT = fmt

    @staticmethod
    def find_primes(start, end, cnt=0):
        """
         Finds all prime numbers between start and end. If cnt is provided, funtion exists
          when number of primes found is equal to cnt

         NOTE : function based on https://www.nayuki.io/page/montgomery-reduction-algorithm
        """
        # Initialize a list
        primes = []
        for possiblePrime in xrange(start, end):
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

        NOTE : function based on https://www.nayuki.io/page/montgomery-reduction-algorithm
        """
        if n < 1:
            assert False, "Number needs to be larger than 1"
        result = {'factors': [], 'exponents': []}
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
                    result['exponents'][-1] += 1
                end = math.sqrt(n)
            i += 1
        if n > 1:
            result['factors'].append(n)
            result['exponents'].append(1)
            while n % i == 0 and n//i > 1:
                n //= i
                result['exponents'][-1] += 1
        if np.prod([r ** e for r, e in zip(result['factors'], result['exponents'])]) != number:
            print( number)
            print( result)
            assert False, "Incorrect factorization"

        return result


