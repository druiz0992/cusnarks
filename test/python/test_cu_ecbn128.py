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
from __future__ import print_function 
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

ECBN128_datafile = './aux_data/ecbn128_data_210.npz'

class CUECTest(unittest.TestCase):
    TEST_ITER = 5000
    curve_data = ZUtils.CURVE_DATA['BN128']
    prime = curve_data['prime_r']
    nsamples = int(1024)
    ntest_points = 6
    u256_p = BigInt(prime).as_uint256()
    ZField(prime, curve_data['curve'])
    ECC.init(curve_data['curve_params'])

    if os.path.exists(ECBN128_datafile):
        npzfile = np.load(ECBN128_datafile)
        #ecbn128_scl   = npzfile['scl']
        #ecbn128_ecjac = npzfile['ecjac']
        #ecbn128_ecjac_rdc = npzfile['ecjac_rdc']
        ecbn128_vector_u256 = npzfile['ecv_u256']
        ecbn128_vector_u256_rdc = npzfile['ecv_u256_rdc']
        ecbn128_scl_u256 = npzfile['scl_u256']
        
        #r_add = npzfile['radd']
        r_add_rdc = npzfile['radd_rdc']
        #r_double = npzfile['rdouble']
        r_double_rdc = npzfile['rdouble_rdc']
        #r_mul = npzfile['rmul']
        r_mul_rdc = npzfile['rmul_rdc']
        #r_mad = npzfile['rmad']
        r_mad_rdc = npzfile['rmad_rdc']
        nsamples = int(len(ecbn128_vector_u256)/ECP_JAC_INDIMS)

        """
        TODO : prepare LUT computing all combinations of Sum Pi, for a window
        ex : w = 5 => P1, P2, P3, P4, P5
         LUT : idx 00000      0
                   00001      P1
                   00010      P2
                   00011      P1 + P2
                   00100      P2
                   00101      P1 + P2
                   ....
                   11111      P1 + P2 + P3 + P4 + P5
        lut_dim = 5
        lut_ec = np.range((1<<lut_dim)-1)
        for i in range(lut_dim):
           lut_ec[(1<<i)-1] = ecbn128_ecjac_rdca[i<<1]
        for i in range(1,1<<lut_dim):
           lut_ec[i] = ecbn128_ecjac_rdca[a] + ecbn128_ecjac_rdca[b]

        """


    else:

        print("Generating Random scalars....",end='')
        ecbn128_scl =      [randint(1,prime-1) for x in xrange(nsamples)]
        print("Done\n")
        print("Converting Random scalars to u256...",end='')
        ecbn128_scl_u256 = [BigInt(x_).as_uint256() for x_ in ecbn128_scl]
        print("Done\n")

        ecbn128_ecjac, ecbn128_ecjac_rdc  = np.asarray(ECC.rand(nsamples, ectype = 1, reduce=True, verbose="Generating Random EC points...\t"))
        print("Done\n")
         
        print("Forming vector...",end='')
        ecbn128_ecjac_u256 = np.asarray([[x.get_P()[0].as_uint256(),
                                          x.get_P()[1].as_uint256(),
                                          x.get_P()[2].as_uint256()] for x in ecbn128_ecjac])
        ecbn128_ecjac_u256_rdc = np.asarray([[x.get_P()[0].as_uint256(),
                                          x.get_P()[1].as_uint256(),
                                          x.get_P()[2].as_uint256()] for x in ecbn128_ecjac_rdc])

        ecbn128_vector_u256 = np.zeros((ECP_JAC_INDIMS*nsamples,NWORDS_256BIT), dtype=np.uint32)
        #ecbn128_vector_u256[::3] = ecbn128_scl_u256
        ecbn128_vector_u256[::2] = ecbn128_ecjac_u256[:,::3].reshape((-1,NWORDS_256BIT))
        ecbn128_vector_u256[1::2] = ecbn128_ecjac_u256[:,1::3].reshape((-1,NWORDS_256BIT))

        ecbn128_vector_u256_rdc = np.zeros((ECP_JAC_INDIMS*nsamples,NWORDS_256BIT), dtype=np.uint32)
        #ecbn128_vector_u256_rdc[::3] = ecbn128_scl_u256
        ecbn128_vector_u256_rdc[::2] = ecbn128_ecjac_u256_rdc[:,::3].reshape((-1,NWORDS_256BIT))
        ecbn128_vector_u256_rdc[1::2] = ecbn128_ecjac_u256_rdc[:,1::3].reshape((-1,NWORDS_256BIT))
        print("Done\n")

        print("Adding EC points...",end='')
        r_add     = [(x + y) for x, y in zip(ecbn128_ecjac[::2], ecbn128_ecjac[1::2])]
        r_add_rdc = [(x + y) for x, y in zip(ecbn128_ecjac_rdc[::2], ecbn128_ecjac_rdc[1::2])]
        r_add_u256 = np.concatenate([[x.get_P()[0].as_uint256(),x.get_P()[1].as_uint256(), x.get_P()[2].as_uint256()] for x in r_add])
        r_add_rdc_u256 = np.concatenate([[x.get_P()[0].as_uint256(),x.get_P()[1].as_uint256(), x.get_P()[2].as_uint256()] for x in r_add_rdc])
        print("Done\n")

        print("Doubling EC points...",end='')
        r_double = [x.double() for x in ecbn128_ecjac]
        r_double_rdc = [x.reduce() for x in r_double]
        r_double_u256 = np.concatenate([[x.get_P()[0].as_uint256(),x.get_P()[1].as_uint256(), x.get_P()[2].as_uint256()] for x in r_double])
        r_double_rdc_u256 = np.concatenate([[x.get_P()[0].as_uint256(),x.get_P()[1].as_uint256(), x.get_P()[2].as_uint256()] for x in r_double_rdc])
        print("Done\n")

        print("Multiplying EC points by scalar...",end='')
        sys.stdout.flush()
        r_mul = [x * scl for x,scl in zip(ecbn128_ecjac,ecbn128_scl)]
        r_mul_rdc = [x.reduce() for x in r_mul]
        r_mul_u256 = np.concatenate([[x.get_P()[0].as_uint256(),x.get_P()[1].as_uint256(), x.get_P()[2].as_uint256()] for x in r_mul])
        r_mul_rdc_u256 = np.concatenate([[x.get_P()[0].as_uint256(),x.get_P()[1].as_uint256(), x.get_P()[2].as_uint256()] for x in r_mul_rdc])
        print("Done\n")

        print("Multiplying/Add EC points...",end='')
        sys.stdout.flush()
        r_mad = np.copy(r_mul)
        r_mad_rdc = np.copy(r_mul_rdc)
        while len(r_mad) != 1:
          r_mad     = [x1 + x2 for x1,x2 in zip(r_mad[::2], r_mad[1::2])]
        r_mad_rdc =  [r_mad[0].reduce()]
        r_mad_u256 = np.concatenate([[x.get_P()[0].as_uint256(),x.get_P()[1].as_uint256(), x.get_P()[2].as_uint256()] for x in r_mad])
        r_mad_rdc_u256 = np.concatenate([[x.get_P()[0].as_uint256(),x.get_P()[1].as_uint256(), x.get_P()[2].as_uint256()] for x in r_mad_rdc])
        print("Done\n")


        print("Saving data...\n",end='')
        np.savez_compressed(ECBN128_datafile, scl=ecbn128_scl, ecjac=ecbn128_ecjac, ecjac_rdc=ecbn128_ecjac_rdc,
                                   ecv_u256=ecbn128_vector_u256, ecv_u256_rdc=ecbn128_vector_u256_rdc, scl_u256=ecbn128_scl_u256,
                                   radd=r_add_u256, radd_rdc=r_add_rdc_u256, rdouble=r_double_u256, rdouble_rdc=r_double_rdc_u256,
                                   rmul=r_mul_u256, rmul_rdc=r_mul_rdc_u256, rmad=r_mad_u256, rmad_rdc=r_mad_rdc_u256)
        print("Done\n")



    def test_0is_on_curve(self):
  
        #ecbn128_pt_ec = CUECTest.ecbn128_ecjac
        ecbn128_pt_ec = np.zeros((ECP_JAC_OUTDIMS*CUECTest.nsamples,NWORDS_256BIT),dtype=np.uint32)
        ecbn128_pt_ec[0::3] = CUECTest.ecbn128_vector_u256[0::2]
        ecbn128_pt_ec[1::3] = CUECTest.ecbn128_vector_u256[1::2]
        ecbn128_pt_ec[2::3] = [ZFieldElExt(1).as_uint256()] * int(len(CUECTest.ecbn128_vector_u256)/ECP_JAC_INDIMS)
        ecbn128_pt_ec = ECC.from_uint256(ecbn128_pt_ec, in_ectype=1, out_ectype=1, reduced=False)
       
        for P in ecbn128_pt_ec:
            self.assertTrue(P.is_on_curve())

        #ecbn128_pt_ec = CUECTest.ecbn128_ecjac_rdc
        ecbn128_pt_ec = np.zeros((ECP_JAC_OUTDIMS*CUECTest.nsamples,NWORDS_256BIT),dtype=np.uint32)
        ecbn128_pt_ec[0::3] = CUECTest.ecbn128_vector_u256_rdc[0::2]
        ecbn128_pt_ec[1::3] = CUECTest.ecbn128_vector_u256_rdc[1::2]
        ecbn128_pt_ec[2::3] = [ZFieldElExt(1).reduce().as_uint256()] * int(len(CUECTest.ecbn128_vector_u256)/ECP_JAC_INDIMS)
        ecbn128_pt_ec = ECC.from_uint256(ecbn128_pt_ec, in_ectype=1, out_ectype=1, reduced=True)

       
        for P in ecbn128_pt_ec:
            self.assertTrue(P.is_on_curve())


    def test_1kernels(self):

        ecbn128_vector = CUECTest.ecbn128_vector_u256_rdc
        ecbn128_vector_full = np.concatenate((CUECTest.ecbn128_scl_u256,CUECTest.ecbn128_vector_u256_rdc))
        nsamples = CUECTest.nsamples
        r_add = CUECTest.r_add_rdc
        r_double = CUECTest.r_double_rdc
        r_mul = CUECTest.r_mul_rdc
        r_mad = CUECTest.r_mad_rdc
        ecbn128 = ECBN128(nsamples, seed=1)

        kernel_config = {'blockD' : [ECBN128_BLOCK_DIM] }
        kernel_params = {'midx' : [MOD_FIELD] ,'premod' : [0], 'in_length' : [nsamples], 'stride' : [1], 'out_length' : nsamples}

        # Test add jacff
        kernel_params['in_length'] = [nsamples  * ECP_JAC_INDIMS]
        kernel_params['out_length']= (nsamples * ECP_JAC_OUTDIMS)/2
        kernel_params['stride'] = [2 * ECP_JAC_INDIMS]
        kernel_config['smemS'] = [0]
        kernel_config['blockD'] = [ECBN128_BLOCK_DIM]
        kernel_config['kernel_idx'] = [CB_EC_JACAFF_ADD]
        kernel_params['padding_idx'] = [0]
        kernel_params['premod'] = [0]
        kernel_params['midx'] = [MOD_FIELD]
        kernel_config['gridD'] = [0]

        result,_ = ecbn128.kernelLaunch(ecbn128_vector, kernel_config, kernel_params )
        self.assertTrue(len(result)/ECP_JAC_OUTDIMS == nsamples/2)
        self.assertTrue(all(np.concatenate(result == r_add)))

        # Test add jac

        kernel_params['in_length'] = [nsamples  * ECP_JAC_OUTDIMS]
        kernel_params['out_length']= (nsamples * ECP_JAC_OUTDIMS)/2
        kernel_params['stride'] = [2 * ECP_JAC_OUTDIMS]
        kernel_config['smemS'] = [0]
        kernel_config['blockD'] = [ECBN128_BLOCK_DIM]
        kernel_config['kernel_idx'] = [CB_EC_JAC_ADD]
        kernel_params['padding_idx'] = [0]
        kernel_params['premod'] = [0]
        kernel_params['midx'] = [MOD_FIELD]
        kernel_config['gridD'] = [0]

        ecbn128_vector_ext = np.zeros((nsamples * ECP_JAC_OUTDIMS, 8),dtype=np.uint32)
        ecbn128_vector_ext[::3] = ecbn128_vector[::2]
        ecbn128_vector_ext[1::3] = ecbn128_vector[1::2]
        ecbn128_vector_ext[2::3] = np.tile(ZFieldElExt(1).reduce().as_uint256(),(nsamples,1))
        result,_ = ecbn128.kernelLaunch(ecbn128_vector_ext, kernel_config, kernel_params )
        self.assertTrue(len(result)/ECP_JAC_OUTDIMS == nsamples/2)
        self.assertTrue(all(np.concatenate(result == r_add)))

        # Test double jacaff
        kernel_params['in_length'] = [nsamples  * ECP_JAC_INDIMS]
        kernel_params['out_length']= (nsamples * ECP_JAC_OUTDIMS)
        kernel_params['stride'] = [1 * ECP_JAC_INDIMS]
        kernel_config['smemS'] = [0]
        kernel_config['blockD'] = [ECBN128_BLOCK_DIM]
        kernel_config['kernel_idx'] = [CB_EC_JACAFF_DOUBLE]
        kernel_params['padding_idx'] = [0]
        kernel_params['premod'] = [0]
        kernel_config['gridD'] = [0]
        kernel_params['midx'] = [MOD_FIELD]

        #ecbn128_vector[::3] = ecbn128_vector[1::3]
        #ecbn128_vector[1::3] = ecbn128_vector[2::3]
        # ecbn128_vector[2::3] = np.tile(ZFieldElExt(1).reduce().as_uint256(),(1024,1))

        result,_ = ecbn128.kernelLaunch(ecbn128_vector, kernel_config, kernel_params )
        #result_ec = ECC.from_uint256(result, in_ectype=1, out_ectype=1, reduced=True)
        self.assertTrue(len(result)/ECP_JAC_OUTDIMS == nsamples)
        self.assertTrue(all(np.concatenate(result == r_double)))

        # Test double jac
        kernel_params['in_length'] = [nsamples  * ECP_JAC_OUTDIMS]
        kernel_params['out_length']= (nsamples * ECP_JAC_OUTDIMS)
        kernel_params['stride'] = [1 * ECP_JAC_OUTDIMS]
        kernel_config['smemS'] = [0]
        kernel_config['blockD'] = [ECBN128_BLOCK_DIM]
        kernel_config['kernel_idx'] = [CB_EC_JAC_DOUBLE]
        kernel_params['padding_idx'] = [0]
        kernel_params['premod'] = [0]
        kernel_config['gridD'] = [0]
        kernel_params['midx'] = [MOD_FIELD]

        ecbn128_vector_ext = np.zeros((nsamples * ECP_JAC_OUTDIMS, 8),dtype=np.uint32)
        ecbn128_vector_ext[::3] = ecbn128_vector[::2]
        ecbn128_vector_ext[1::3] = ecbn128_vector[1::2]
        ecbn128_vector_ext[2::3] = np.tile(ZFieldElExt(1).reduce().as_uint256(),(nsamples,1))
        result,_ = ecbn128.kernelLaunch(ecbn128_vector_ext, kernel_config, kernel_params )
        #result_ec = ECC.from_uint256(result, in_ectype=1, out_ectype=1, reduced=True)
        self.assertTrue(len(result)/ECP_JAC_OUTDIMS == nsamples)
        self.assertTrue(all(np.concatenate(result == r_double)))

        # Test sc mul jac
        kernel_params['in_length'] = [nsamples  * (ECP_JAC_INDIMS+U256_NDIMS)]
        kernel_params['out_length']= (nsamples * ECP_JAC_OUTDIMS)
        kernel_params['stride'] = [1 * (ECP_JAC_INDIMS+U256_NDIMS)]
        kernel_config['smemS'] = [0]
        kernel_config['blockD'] = [ECBN128_BLOCK_DIM]
        kernel_config['kernel_idx'] = [CB_EC_JAC_MUL]
        kernel_params['padding_idx'] = [0]
        kernel_params['premod'] = [0]
        kernel_params['midx'] = [MOD_FIELD]
        kernel_config['gridD'] = [0]

        result,_ = ecbn128.kernelLaunch(ecbn128_vector_full, kernel_config, kernel_params )
        #result_ec = ECC.from_uint256(result, in_ectype=1, out_ectype=1, reduced=True)
        self.assertTrue(len(result)/ECP_JAC_OUTDIMS == nsamples)
        self.assertTrue(ECC.from_uint256(result,in_ectype=1, out_ectype=1, reduced=True) ==
                              ECC.from_uint256(r_mul, in_ectype=1, out_ectype=1, reduced=True))

        scl_v = BigInt.from_uint256(CUECTest.ecbn128_scl_u256[0])
        ec_v = np.zeros((6,8),dtype=np.uint32)
        ec_v[0:2] = CUECTest.ecbn128_vector_u256_rdc[0:2]
        ec_v[3:5] = CUECTest.ecbn128_vector_u256_rdc[3:5]
        ec_v[2] = ZFieldElExt(1).reduce().as_uint256()
        ec_v[5] = ZFieldElExt(1).reduce().as_uint256()
        ec_v2  = ECC.from_uint256(ec_v, in_ectype=1, out_ectype=1, reduced=True)
        ec_r = scl_v * ec_v2[0]

        # Test mad jac
        kernel_params['stride'] = [ECP_JAC_OUTDIMS, ECP_JAC_OUTDIMS]
        kernel_config['blockD'] = [64,64]
        kernel_params['premul'] = [1,0]
        kernel_config['gridD'] = [0,1]
        kernel_config['smemS'] = [kernel_config['blockD'][0] * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4, \
                                      kernel_config['blockD'][1] * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4]
        kernel_config['kernel_idx'] = [CB_EC_JAC_MAD, CB_EC_JAC_MAD]
        out_len1 = ECP_JAC_OUTDIMS * ((nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS) -1) /
                                          (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS))
        kernel_params['in_length'] = [nsamples * (ECP_JAC_INDIMS+U256_NDIMS), out_len1]
        kernel_params['out_length'] = 1 * ECP_JAC_OUTDIMS
        kernel_params['padding_idx'] = [0,0]
        kernel_params['premod'] = [0,0]
        kernel_params['midx'] = [MOD_FIELD, MOD_FIELD]
        min_length = [ECP_JAC_OUTDIMS * \
                    (kernel_config['blockD'][idx] * kernel_params['stride'][idx]/ECP_JAC_OUTDIMS) for idx in range(len(kernel_params['stride']))]

        ecbn128_vector_mad = np.copy(ecbn128_vector_full)
        for bidx, l in enumerate(kernel_params['in_length']):
           if l < min_length[bidx]:
              if bidx == 0:
                 zeros = np.zeros((min_length[bidx] - kernel_params['in_length'][bidx],NWORDS_256BIT), dtype=np.uint32)
                 ecbn128_vector_mad = np.concatenate((ecbn128_vector_full,zeros))
                 kernel_params['in_length'][bidx] = min_length[bidx]
              else:
                 kernel_params['in_length'][bidx] = min_length[bidx]
                 kernel_params['padding_idx'][bidx] = l/ECP_JAC_OUTDIMS

        result,_ = ecbn128.kernelLaunch(ecbn128_vector_mad, kernel_config, kernel_params, 2 )

        #a = ZFieldElExt(BigInt.from_uint256(ecbn128_vector[0]))
        #x1 = ZFieldElRedc(BigInt.from_uint256(ecbn128_vector[1]))
        #y1 = ZFieldElRedc(BigInt.from_uint256(ecbn128_vector[2]))
        #P1 = ECCJacobian([x1,y1,ZFieldElExt(1).reduce()])
        #P2 = a * P1

        #debugReducedECCAdd(r_mul, result, result2)

        # I need to convert to EC point, as u256 representation can be different for same EC point
        result_ec = ECC.from_uint256(result, in_ectype=1, out_ectype=1, reduced=True)
        r_mad_ec = ECC.from_uint256(r_mad, in_ectype=1, out_ectype=1, reduced=True)
        #self.assertTrue(len(result) == kernel_params['out_length'])
        #self.assertTrue(result_ec == r_mad_ec)

        for niter in xrange(CUECTest.TEST_ITER):
            # Test mad jac shuffle
            kernel_params['stride'] = [ECP_JAC_INDIMS + U256_NDIMS, ECP_JAC_OUTDIMS, ECP_JAC_OUTDIMS]
            kernel_config['blockD'] = [256, 256,32]
            kernel_params['premul'] = [0,0,0]
            kernel_params['premod'] = [0,0,0]
            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD, MOD_FIELD]
            kernel_config['smemS'] = [0,
                                      kernel_config['blockD'][1]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4,
                                      kernel_config['blockD'][2]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4]
            kernel_config['kernel_idx'] = [CB_EC_JAC_MUL, CB_EC_JAC_MAD_SHFL, CB_EC_JAC_MAD_SHFL]
            out_len1 = ECP_JAC_OUTDIMS * ((nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS) -1) /
                                          (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS))
            kernel_params['in_length'] = [nsamples * (ECP_JAC_INDIMS +U256_NDIMS), nsamples * (ECP_JAC_OUTDIMS), out_len1]
            kernel_params['out_length'] = 1 * ECP_JAC_OUTDIMS
            kernel_params['padding_idx'] = [0,0,0]
            kernel_config['gridD'] = [0,0,1]
            min_length = [ECP_JAC_OUTDIMS * \
                    (kernel_config['blockD'][idx] * kernel_params['stride'][idx]/ECP_JAC_OUTDIMS) for idx in range(len(kernel_params['stride']))]
            nkernels = 3

            idx_v = sortu256_idx_h(ecbn128_vector_mad[:nsamples])
            sorted_scl_vector = ecbn128_vector_mad[:nsamples][idx_v]
            tmp_v = np.reshape(ecbn128_vector_mad[nsamples:],(-1,2,8))
            sorted_ecc_vector = np.reshape(tmp_v[idx_v],(-1,8))
            sorted_ecbn128_vector_mad = np.concatenate((sorted_scl_vector,sorted_ecc_vector))

            result,_ = ecbn128.kernelLaunch(sorted_ecbn128_vector_mad, kernel_config, kernel_params,nkernels )

            result2 = 0
            #debugReducedECCAddShfl(r_mul, result, result2)

            # I need to convert to EC point, as u256 representation can be different for same EC point
            result_ec = ECC.from_uint256(result, in_ectype=1, out_ectype=1, reduced=True)
            r_mad_ec = ECC.from_uint256(r_mad, in_ectype=1, out_ectype=1, reduced=True)
            self.assertTrue(len(result) == kernel_params['out_length'])
            self.assertTrue(result_ec == r_mad_ec)

def debugReducedECCAddShfl(r_mul, result1, result2):

   tt = ECC.from_uint256(r_mul,in_ectype=1, out_ectype=1, reduced=True)
   # 32 -> 16 
   for i in range(len(tt)/32):
      tt[16*i:16*(i+1)] = [x + y for x,y in zip(tt[2*i*16:2*i*16+16],tt[2*i*16+16:2*i*16+32])]
   tt = tt[:len(tt)/2]
   # 16 -> 8 
   for i in range(len(tt)/16):
      tt[8*i:8*(i+1)] = [x + y for x,y in zip(tt[2*i*8:2*i*8+8],tt[2*i*8+8:2*i*8+16])]
   tt = tt[:len(tt)/2]
   # 8 -> 4 
   for i in range(len(tt)/8):
      tt[4*i:4*(i+1)] = [x + y for x,y in zip(tt[2*i*4:2*i*4+4],tt[2*i*4+4:2*i*4+8])]
   tt = tt[:len(tt)/2]
   # 4 -> 2 
   for i in range(len(tt)/4):
      tt[2*i:2*(i+1)] = [x + y for x,y in zip(tt[2*i*2:2*i*2+2],tt[2*i*2+2:2*i*2+4])]
   tt = tt[:len(tt)/2]
   # 2 -> 1 
   for i in range(len(tt)/2):
      tt[i:(i+1)] = [x + y for x,y in zip(tt[2*i:2*i+1],tt[2*i+1:2*i+2])]
   tt = tt[:len(tt)/2]

   # 8 -> 4 
   for i in range(len(tt)/8):
      tt[4*i:4*(i+1)] = [x + y for x,y in zip(tt[2*i*4:2*i*4+4],tt[2*i*4+4:2*i*4+8])]
   tt = tt[:len(tt)/2]
   # 4 -> 2 
   for i in range(len(tt)/4):
      tt[2*i:2*(i+1)] = [x + y for x,y in zip(tt[2*i*2:2*i*2+2],tt[2*i*2+2:2*i*2+4])]
   tt = tt[:len(tt)/2]
   # 2 -> 1 
   for i in range(len(tt)/2):
      tt[i:(i+1)] = [x + y for x,y in zip(tt[2*i:2*i+1],tt[2*i+1:2*i+2])]
   tt = tt[:len(tt)/2]


   # stride reduction
   tt = [x + y for x,y in zip(tt[::2],tt[1::2])]
   # 64 -> 32
   for i in range(len(tt)/64):
      tt[32*i:32*(i+1)] = [x + y for x,y in zip(tt[2*i*32:2*i*32+32],tt[2*i*32+32:2*i*32+64])]
   tt = tt[:len(tt)/2]
   # 32 -> 16
   for i in range(len(tt)/32):
      tt[16*i:16*(i+1)] = [x + y for x,y in zip(tt[2*i*16:2*i*16+16],tt[2*i*16+16:2*i*16+32])]
   tt = tt[:len(tt)/2]
   # 16 -> 8 
   for i in range(len(tt)/16):
      tt[8*i:8*(i+1)] = [x + y for x,y in zip(tt[2*i*8:2*i*8+8],tt[2*i*8+8:2*i*8+16])]
   tt = tt[:len(tt)/2]
   # 8 -> 4 
   for i in range(len(tt)/8):
      tt[4*i:4*(i+1)] = [x + y for x,y in zip(tt[2*i*4:2*i*4+4],tt[2*i*4+4:2*i*4+8])]
   tt = tt[:len(tt)/2]
   # 4 -> 2 
   for i in range(len(tt)/4):
      tt[2*i:2*(i+1)] = [x + y for x,y in zip(tt[2*i*2:2*i*2+2],tt[2*i*2+2:2*i*2+4])]
   tt = tt[:len(tt)/2]
   # 2 -> 1 
   for i in range(len(tt)/2):
      tt[i:(i+1)] = [x + y for x,y in zip(tt[2*i:2*i+1],tt[2*i+1:2*i+2])]
   tt = tt[:len(tt)/2]
           

def debugReducedECCAdd(r_mul, result1, result2):

   tt = ECC.from_uint256(r_mul,in_ectype=1, out_ectype=1, reduced=True)
   # stride reduction
   tt = [x + y for x,y in zip(tt[::2],tt[1::2])]
   # 64 -> 32
   for i in range(len(tt)/64):
      tt[32*i:32*(i+1)] = [x + y for x,y in zip(tt[2*i*32:2*i*32+32],tt[2*i*32+32:2*i*32+64])]
   tt = tt[:len(tt)/2]
   # 32 -> 16 
   for i in range(len(tt)/32):
      tt[16*i:16*(i+1)] = [x + y for x,y in zip(tt[2*i*16:2*i*16+16],tt[2*i*16+16:2*i*16+32])]
   tt = tt[:len(tt)/2]
   # 16 -> 8 
   for i in range(len(tt)/16):
      tt[8*i:8*(i+1)] = [x + y for x,y in zip(tt[2*i*8:2*i*8+8],tt[2*i*8+8:2*i*8+16])]
   tt = tt[:len(tt)/2]
   # 8 -> 4 
   for i in range(len(tt)/8):
      tt[4*i:4*(i+1)] = [x + y for x,y in zip(tt[2*i*4:2*i*4+4],tt[2*i*4+4:2*i*4+8])]
   tt = tt[:len(tt)/2]
   # 4 -> 2 
   for i in range(len(tt)/4):
      tt[2*i:2*(i+1)] = [x + y for x,y in zip(tt[2*i*2:2*i*2+2],tt[2*i*2+2:2*i*2+4])]
   tt = tt[:len(tt)/2]
   # 2 -> 1 
   for i in range(len(tt)/2):
      tt[i:(i+1)] = [x + y for x,y in zip(tt[2*i:2*i+1],tt[2*i+1:2*i+2])]
   tt = tt[:len(tt)/2]


   tt = ECC.from_uint256(result,in_ectype=1, out_ectype=1, reduced=True)
   # stride reduction
   tt = [x + y for x,y in zip(tt[::2],tt[1::2])]
   # 64 -> 32
   for i in range(len(tt)/64):
      tt[32*i:32*(i+1)] = [x + y for x,y in zip(tt[2*i*32:2*i*32+32],tt[2*i*32+32:2*i*32+64])]
   tt = tt[:len(tt)/2]
   # 32 -> 16
   for i in range(len(tt)/32):
      tt[16*i:16*(i+1)] = [x + y for x,y in zip(tt[2*i*16:2*i*16+16],tt[2*i*16+16:2*i*16+32])]
   tt = tt[:len(tt)/2]
   # 16 -> 8 
   for i in range(len(tt)/16):
      tt[8*i:8*(i+1)] = [x + y for x,y in zip(tt[2*i*8:2*i*8+8],tt[2*i*8+8:2*i*8+16])]
   tt = tt[:len(tt)/2]
   # 8 -> 4 
   for i in range(len(tt)/8):
      tt[4*i:4*(i+1)] = [x + y for x,y in zip(tt[2*i*4:2*i*4+4],tt[2*i*4+4:2*i*4+8])]
   tt = tt[:len(tt)/2]
   # 4 -> 2 
   for i in range(len(tt)/4):
      tt[2*i:2*(i+1)] = [x + y for x,y in zip(tt[2*i*2:2*i*2+2],tt[2*i*2+2:2*i*2+4])]
   tt = tt[:len(tt)/2]
   # 2 -> 1 
   for i in range(len(tt)/2):
      tt[i:(i+1)] = [x + y for x,y in zip(tt[2*i:2*i+1],tt[2*i+1:2*i+2])]
   tt = tt[:len(tt)/2]

if __name__ == "__main__":
    unittest.main()
