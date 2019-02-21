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
// File name  : test_cu_ecbn128.py
//
// Date       : 19/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   ECBN128 CUDA test
//
// ------------------------------------------------------------------

"""
import os,sys, os.path
import unittest
import numpy as np
from random import randint, sample

sys.path.append('../../src/python')

from bigint import *
from zutils import *
from zfield import *
from constants import *
from ecc import *


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False


sys.path.append('../../src/python')
from bigint import *

ECBN128_datafile = './aux_data/ecbn128_data.npz'

class CUECTest(unittest.TestCase):
    TEST_ITER = 1000
    curve_data = ZUtils.CURVE_DATA['BN128']
    prime = curve_data['prime_r']
    nsamples = 1024*128
    ntest_points = 6
    u256_p = BigInt(prime).as_uint256()
    if use_pycusnarks:
       ecbn128 = ECBN128(nsamples, seed=1)
    ZField(prime, curve_data['curve'])
    ECC.init(curve_data['curve_params'])

    #if use_pycusnarks:
     #  ecbn128_scalars = ecbn128.rand(nsamples)
    #else :
    if os.path.exists(ECBN128_datafile):
        npzfile = np.load(ECBN128_datafile)
        ecbn128_scl   = npzfile['scl']
        ecbn128_ecaff = npzfile['ecaff']
        ecbn128_ecjac = npzfile['ecjac']
        ecbn128_vector_u256 = npzfile['ecv_256']
    else:

        ecbn128_scl =      [randint(1,prime-1) for x in xrange(nsamples)]
        ecbn128_scl_u256 = [BigInt(x_).as_uint256() for x_ in ecbn128_scl]

        ecbn128_ecaff      = np.asarray(ECC.rand(nsamples, ectype = 2, reduce=True))
        ecbn128_ecjac      = np.asarray([x.to_jacobian() for x in ecbn128_ecaff])

        ecbn128_ecaff_u256 = np.asarray([[x.get_P()[0].as_uint256(),
                                          x.get_P()[1].as_uint256(),
                                          x.get_P()[2].as_uint256()] for x in ecbn128_ecaff])
        ecbn128_vector_u256 = np.zeros((3*nsamples,NWORDS_256BIT), dtype=np.uint32)
        ecbn128_vector_u256[::3] = ecbn128_scl_u256
        ecbn128_vector_u256[1::3] = ecbn128_ecaff_u256[:,::3].reshape((-1,NWORDS_256BIT))
        ecbn128_vector_u256[2::3] = ecbn128_ecaff_u256[:,1::3].reshape((-1,NWORDS_256BIT))

        np.savez(ECBN128_datafile, scl=ecbn128_scl, ecaff=ecbn128_ecaff, ecjac=ecbn128_ecjac, ecv_u256=ecbn128_vector_u256)

    def test_0is_on_curve(self):
  
        ecbn128_pt_ec = CUECTest.ecbn128_pt_ec
       
        for P in ecbn128_pt_ec:
            self.assertTrue(P.is_on_curve())

    def test_1kernels(self):
        if use_pycusnarks:
           ecbn128 = CUECTest.ecbn128

        ecbn128_vector = CUECTest.ecbn128_vector
        ntest_points = CUECTest.ntest_points
        nsamples = CUECTest.nsamples
        ecbn128_pt_ec = CUECTest.ecbn128_pt_ec

        kernel_config = {'blockD' : ECBN128_BLOCK_DIM }
        kernel_params = {'midx' : MOD_FIELD ,'premod' : 1, 'in_length' : nsamples, 'stride' : 1, 'out_length' : nsamples}
        for iter in xrange(CUECTest.TEST_ITER):

            # Test add jac
            test_points = sample(xrange(nsamples/2-2), ntest_points)
            test_points2 = np.multiply(test_points,2)
            ecbn128_pt_ec_vector1 = ecbn128_pt_ec[test_points2]
            ecbn128_pt_ec_vector2 = ecbn128_pt_ec[test_points2+1]
            #ecbn128_pt_ec_vector1 = ecbn128_pt_ec[::2]
            #ecbn128_pt_ec_vector2 = ecbn128_pt_ec[1::2]

            kernel_params['in_length'] = nsamples  * ECK_JAC_INDIMS
            kernel_params['out_length']= (nsamples * ECK_JAC_OUTDIMS)/2
            kernel_params['stride'] = 2
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = ECBN128_BLOCK_DIM

            result = ecbn128.kernelLaunch(CB_EC_JAC_ADD, ecbn128_vector, kernel_config, kernel_params )
            result_ec = ECC.from_uint256(result, in_ectype=1, out_ectype=1, reduced=True)
            r_add = [(x.to_jacobian() + y.to_jacobian()) for x, y in zip(ecbn128_pt_ec_vector1, ecbn128_pt_ec_vector2)]
            self.assertTrue(len(result_ec) == CUECTest.nsamples/2)
            self.assertTrue(all(np.asarray(result_ec)[test_points] == r_add))

            # Test double jac
            test_points = sample(xrange(nsamples-1), ntest_points)
            ecbn128_pt_ec_vector = ecbn128_pt_ec[test_points]
            #ecbn128_pt_ec_vector = ecbn128_pt_ec
            kernel_params['in_length'] = nsamples  * ECK_JAC_INDIMS
            kernel_params['out_length']= (nsamples * ECK_JAC_OUTDIMS)
            kernel_params['stride'] = 1
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = ECBN128_BLOCK_DIM

            result = ecbn128.kernelLaunch(CB_EC_JAC_DOUBLE, ecbn128_vector, kernel_config, kernel_params )
            result_ec = ECC.from_uint256(result, in_ectype=1, out_ectype=1, reduced=True)
            r_double = [x.to_jacobian().double() for x in ecbn128_pt_ec_vector]
            self.assertTrue(len(result_ec) == CUECTest.nsamples)
            self.assertTrue(all(np.asarray(result_ec)[test_points] == r_double))

            # Test sc mul jac
            test_points = sample(xrange(nsamples-1), ntest_points)
            ecbn128_pt_ec_vector = ecbn128_pt_ec[test_points]
            scl_vector = ecbn128_vector[::3][test_points]
            #ecbn128_pt_ec_vector = ecbn128_pt_ec
            #scl_vector = ecbn128_vector[::3]
            #ecbn128_pt_ec_vector[0] = ecbn128_pt_ec_vector[1]
            #scl_vector[1] = scl_vector[1]
            #ecbn128_vector[0:3] = ecbn128_vector[3:6]

            kernel_params['in_length'] = nsamples  * ECK_JAC_INDIMS
            kernel_params['out_length']= (nsamples * ECK_JAC_OUTDIMS)
            kernel_params['stride'] = 1
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = ECBN128_BLOCK_DIM

            result = ecbn128.kernelLaunch(CB_EC_JAC_MUL, ecbn128_vector, kernel_config, kernel_params )
            result_ec = ECC.from_uint256(result, in_ectype=1, out_ectype=1, reduced=True)
            r_mul = [x.to_jacobian() * BigInt.from_uint256(scl).as_long() for x,scl in zip(ecbn128_pt_ec_vector,scl_vector)]
            self.assertTrue(len(result_ec) == CUECTest.nsamples)
            self.assertTrue(all(np.asarray(result_ec)[test_points] == r_mul))


            # Test mac jac
            test_points = sample(xrange(nsamples-1), ntest_points)
            ecbn128_pt_ec_vector = ecbn128_pt_ec[test_points]
            scl_vector = ecbn128_vector[::3][test_points]
            r_mad = [x.to_jacobian() * BigInt.from_uint256(scl).as_long() for x,scl in zip(ecbn128_pt_ec_vector,scl_vector)]
            r_mad = [x + x for x in r_mad]
            kernel_params['in_length'] = nsamples
            kernel_params['stride'] = 4
            kernel_config['blockD'] = ECBN128_BLOCK_DIM 
            kernel_params['out_length'] = (nsamples + (kernel_config['blockD']*kernel_params['stride']) -1) / (kernel_config['blockD']*kernel_params['stride'])
            kernel_config['smemS'] = kernel_config['blockD'] * NWORDS_256BIT * ECK_JAC_OUTDIMS * 4 
            result = ecbn128.kernelLaunch(CB_EC_JAC_MAD, ecbn128_vector, kernel_config, kernel_params )

            kernel_params['in_length'] = kernel_params['out_length']
            kernel_params['stride'] = 2
            kernel_params['out_length'] = 1
            kernel_config['blockD'] = 64
            kernel_config['smemS'] = kernel_config['blockD'] * NWORDS_256BIT * ECK_JAC_OUTDIMS * 4 
            min_length = kernel_config['blockD'] * kernel_params['stride']
            if kernel_params['in_length'] < min_length:
               zeros = np.zeros((min_length - kernel_params['in_length'],NWORDS_256BIT * ECK_JAC_OUTDIMS), dtype=np.uint32)
               result = np.concatenate((result,zeros))
               kernel_params['in_length'] = min_length

            result = ecbn128.kernelLaunch(CB_EC_JAC_MAD, result, kernel_config, kernel_params )
    
            self.assertTrue(len(result) == kernel_params['out_length'])
            self.assertTrue(all(np.concatenate(result == r_mad)))


if __name__ == "__main__":
    unittest.main()
