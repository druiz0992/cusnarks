#!/usr/bin/python
 
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
// File name  : groth_prover
//
// Date       : 27/01/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Zero Kowledge Groth prover implementation
# 


// Description:
//    
//   TODO
//    
// ------------------------------------------------------------------

"""
import json,ast
import os.path
import numpy as np
import time

from zutils import ZUtils
from random import randint
from zfield import *
from ecc import *
from zpoly import *
from constants import *
from pysnarks_utils import *
from cuda_wrapper import *

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

#Input files
DEFAULT_WITNESS_LOC = "../../data/witness_pedersen.json"
DEFAULT_PROVING_KEY_LOC = "../../data/proving_key_pedersen.json"
#DEFAULT_WITNESS_LOC = "../../data/witness_multiplier.json"
#DEFAULT_PROVING_KEY_LOC = "../../data/proving_key_multiplier.json"
#Output files
DEFAULT_PROOF_LOC =  "../../data/proof.bin"
DEFAULT_PUBLIC_LOC =  "../../data/public.bin"
DATA_FOLDER = "../../data/"

ROOTS_1M_filename = '../../data/zpoly_roots_1M.bin'

class GrothProver(object):
    
    GroupIDX = 0
    FieldIDX = 1

    def __init__(self, proving_key_f, curve='BN128',  out_proof_f = DEFAULT_PROOF_LOC, out_public_f = DEFAULT_PUBLIC_LOC, in_point_fmt=ZUtils.FEXT, in_coor_fmt = ZUtils.JACOBIAN, accel = True, inrand=False):

        self.accel = accel

        self.out_proof_f = out_proof_f
        self.out_public_f = out_public_f
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data)
        ZPoly.init(GrothProver.FieldIDX)

        self.inrand = inrand

        data_init = self.load_pkdata(proving_key_f)
        if not data_init:
          #init class variables
          self.init_vars()

          if use_pycusnarks and self.accel:
             self.init_u256_vars(proving_key_f)

        if use_pycusnarks and self.accel:
          self.roots1M_rdc_u256 = readU256DataFile_h(ROOTS_1M_filename.encode("UTF-8"), 1<<20, 1<<20)

          self.nVars = self.vk_proof['nVars']
          self.batch_size = 1<<20
          self.ecbn128 = ECBN128(self.batch_size + 2,   seed=1)
          self.ec2bn128 = EC2BN128(self.batch_size + 2, seed=1)
          self.cuzpoly = ZCUPoly(self.batch_size, seed=1)
          
    def read_witness_data(self, witness_f):
       ## Open and parse witness data
       if os.path.isfile(witness_f):
           f = open(witness_f,'r')
           self.witness_scl = [BigInt(c) for c in ast.literal_eval(json.dumps(json.load(f)))]
           f.close()
       else:
          print("File doesn't exist")
          assert False
       
    def load_pkdata(self, proving_key_f):
       proving_key_fnpz = proving_key_f[:-4] + 'npz'
       data_init = False
       if use_pycusnarks and os.path.isfile(proving_key_fnpz) and self.accel:
          npzfile = np.load(proving_key_fnpz)
          self.vk_alfa_1_eccf1_u256 = npzfile['alfa_1_u256']
          self.vk_beta_1_eccf1_u256 = npzfile['beta_1_u256']
          self.vk_delta_1_eccf1_u256= npzfile['delta_1_u256']
          self.vk_beta_2_eccf2_u256 = npzfile['beta_2_u256']
          self.vk_delta_2_eccf2_u256 = npzfile['delta_2_u256']

          self.A_eccf1_u256 = npzfile['A_u256']
          self.B1_eccf1_u256 = npzfile['B1_u256']
          self.B2_eccf2_u256 = npzfile['B2_u256']
          self.C_eccf1_u256  = npzfile['C_u256']
          self.hExps_eccf1_u256 = npzfile['hExps_u256']
          self.polsA_sps_u256 = npzfile['polsA_u256']
          self.polsB_sps_u256 = npzfile['polsB_u256']
          self.polsC_sps_u256 = npzfile['polsC_u256']
          self.vk_proof={}
          self.vk_proof['nVars'] = npzfile['nvars']
          self.vk_proof['nPublic'] = npzfile['npublic']
          self.vk_proof['domainSize'] = npzfile['domain_size']
          self.vk_proof['domainBits'] = npzfile['domain_bits']
          data_init = True
       ## Open and parse proving key data
       elif os.path.isfile(proving_key_f):
           f = open(proving_key_f,'r')
           tmp_data = json.load(f)
           self.vk_proof = json_to_dict(tmp_data)
           f.close()
       else:
          print("File doesn't exist")
          assert False


       return data_init

    def init_u256_vars(self, proving_key_f):
       ZField.set_field(GrothProver.GroupIDX)
       self.vk_alfa_1_eccf1_u256 = ECC.as_uint256(self.vk_alfa_1_eccf1,remove_last=True, as_reduced=True)
       self.vk_beta_1_eccf1_u256 = ECC.as_uint256(self.vk_beta_1_eccf1,remove_last=True, as_reduced=True )
       self.vk_delta_1_eccf1_u256= ECC.as_uint256(self.vk_delta_1_eccf1,remove_last= True, as_reduced=True)
       self.vk_beta_2_eccf2_u256 = ECC.as_uint256(self.vk_beta_2_eccf2, remove_last = True, as_reduced=True)
       self.vk_delta_2_eccf2_u256 = ECC.as_uint256(self.vk_delta_2_eccf2, remove_last = True, as_reduced=True)

       self.A_eccf1_u256 = ECC.as_uint256(self.A_eccf1, remove_last = True, as_reduced=True)
       self.B1_eccf1_u256 = ECC.as_uint256(self.B1_eccf1, remove_last = True, as_reduced=True)
       self.B2_eccf2_u256 = ECC.as_uint256(self.B2_eccf2, remove_last = True, as_reduced=True)
       self.C_eccf1_u256  = ECC.as_uint256(self.C_eccf1, remove_last=True, as_reduced=True)
       self.hExps_eccf1_u256 = ECC.as_uint256(self.hExps_eccf1, remove_last=True, as_reduced=True)



       ZField.set_field(GrothProver.FieldIDX)
       polsA_l = []
       polsA_p = []
       for pol in self.polsA_sps:
         l,p = pol.reduce().as_uint256() 
         polsA_l.append(l)
         polsA_p.append(p)
       self.polsA_sps_u256 = np.asarray(np.concatenate((np.asarray([len(polsA_l)]),
                                    np.concatenate((np.cumsum(polsA_l),np.concatenate(polsA_p))))),dtype=np.uint32)

       polsB_l = []
       polsB_p = []
       for pol in self.polsB_sps:
         l,p = pol.reduce().as_uint256() 
         polsB_l.append(l)
         polsB_p.append(p)
       self.polsB_sps_u256 = np.asarray(np.concatenate((np.asarray([len(polsB_l)]),
                                    np.concatenate((np.cumsum(polsB_l),np.concatenate(polsB_p))))),dtype=np.uint32)

       polsC_l = []
       polsC_p = []
       for pol in self.polsC_sps:
         l,p = pol.reduce().as_uint256() 
         polsC_l.append(l)
         polsC_p.append(p)
       self.polsC_sps_u256 = np.asarray(np.concatenate((np.asarray([len(polsC_l)]),
                                    np.concatenate((np.cumsum(polsC_l),np.concatenate(polsC_p))))),dtype=np.uint32)

       
       proving_key_fnpz = proving_key_f[:-4] + 'npz'

       np.savez_compressed(proving_key_fnpz, alfa_1_u256 =  self.vk_alfa_1_eccf1_u256,
                             beta_1_u256 = self.vk_beta_1_eccf1_u256, delta_1_u256 = self.vk_delta_1_eccf1_u256,
                             beta_2_u256 = self.vk_beta_2_eccf2_u256, delta_2_u256 = self.vk_delta_2_eccf2_u256,
                             A_u256 = self.A_eccf1_u256, B1_u256=self.B1_eccf1_u256, B2_u256 = self.B2_eccf2_u256,
                             C_u256 = self.C_eccf1_u256, hExps_u256 =self.hExps_eccf1_u256,
                             polsA_u256 = self.polsA_sps_u256, polsB_u256 = self.polsB_sps_u256, polsC_u256 = self.polsC_sps_u256,
                             nvars = self.vk_proof['nVars'], npublic=self.vk_proof['nPublic'], domain_bits=self.vk_proof['domainBits'],
                             domain_size = self.vk_proof['domainSize'])


    def get_invpoly_u256(self, n):
       fidx = ZField.get_field()

       ZField.set_field(GrothProver.FieldIDX)
       inv_fnpz = DATA_FOLDER + "inv_"+str(n)+".npz"
       if not os.path.isfile(inv_fnpz):
          poly = ZPoly([ZFieldElExt(-1)] + [ZFieldElExt(0) for i in range(n-1)] + [ZFieldElExt(1)])
          nd = (1<<  int(math.ceil(math.log(n+1, 2))) ) -1 - n
          poly = poly.scale(nd)
          inv_poly = poly.inv()
          inv_poly_u256 = inv_poly.as_uint256()

          np.savez_compressed(inv_fnpz, invpoly_data=inv_poly_u256)
       else:  
          npzfile = np.load(inv_fnpz)
          inv_poly_u256 = npzfile['invpoly_data']

       ZField.set_field(fidx)

       return inv_poly_u256
 
    def init_vars(self):
        # Init witness to Field El.
        # TODO :  I am assuming that all field el are FielElExt (witness_scl, polsA_sps, polsB_sps, polsC_sps, alfa1...
        # Witness is initialized a BitInt as it will operate on different fields
        #self.witness_scl = [BigInt(el) for el in self.witness_scl]

        # Init pi's
        self.pi_a_eccf1 = ECC_F1()
        self.pi_b_eccf2 = ECC_F2()
        self.pi_c_eccf1 = ECC_F1()

        self.vk_alfa_1_eccf1 = ECC_F1(p=self.vk_proof['vk_alfa_1'])
        self.vk_beta_1_eccf1 = ECC_F1(p=self.vk_proof['vk_beta_1'])
        self.vk_delta_1_eccf1 = ECC_F1(p=self.vk_proof['vk_delta_1'])

        beta2 = [Z2FieldEl(el) for el in self.vk_proof['vk_beta_2']]
        self.vk_beta_2_eccf2 = ECC_F2(beta2)
        delta2 = [Z2FieldEl(el) for el in self.vk_proof['vk_delta_2']]
        self.vk_delta_2_eccf2 = ECC_F2(delta2)


        self.public_signals = None

        self.A_eccf1     = [ECC_F1(p) for p in self.vk_proof['A']]
        self.B1_eccf1    = [ECC_F1(p) for p in self.vk_proof['B1']]
        self.B2_eccf2    = [ECC_F2(p) for p in self.vk_proof['B2']]
        self.C_eccf1     = [ECC_F1(p) for p in self.vk_proof['C']]
        self.hExps_eccf1 = [ECC_F1(p) for p in self.vk_proof['hExps']]


        ZField.set_field(GrothProver.FieldIDX)

        ZField.find_roots(ZUtils.NROOTS)
        # TODO : This representation may not be optimum. I only have good representation of sparse polynomial,
        #  but not of array of sparse poly (it is also sparse). I should encode it as a dictionary as wekk
        self.polsA_sps = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in self.vk_proof['polsA']]
        self.polsB_sps = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in self.vk_proof['polsB']]
        self.polsC_sps = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in self.vk_proof['polsC']]


    def gen_proof(self, witness_f):
        """
          public_signals, pi_a_eccf1, pi_b_eccf2, pi_c_eccf1 
        """
        # Read witness
        self.t_GP = []
        start = time.time()
        self.read_witness_data(witness_f)
        end = time.time()
        self.t_GP.append(end - start)

        ZField.set_field(GrothProver.FieldIDX)
        # Init r and s scalars
        if self.inrand:
          self.r_scl = BigInt(16261132245285695825038220026199411981758970481965022651127483492661266665974)
          self.s_scl = BigInt(54329550134654175536209911923112271011239733092116208039419485066137536819441)
        else:
          self.r_scl = BigInt(randint(1,ZField.get_extended_p().as_long()-1))
          self.s_scl = BigInt(randint(1,ZField.get_extended_p().as_long()-1))

        if use_pycusnarks and self.accel:
           self.r_scl_u256 = self.r_scl.as_uint256()
           self.s_scl_u256 = self.s_scl.as_uint256()
           self.witness_scl_u256 = \
             np.reshape(np.asarray([el.as_uint256() for el in self.witness_scl], dtype=np.uint32),(-1,NWORDS_256BIT))
           nVars = self.nVars
           nPublic = self.vk_proof['nPublic']
           self.sorted_witness1_idx = sortu256_idx_h(self.witness_scl_u256[:nVars])
           self.sorted_witness2_idx = sortu256_idx_h(self.witness_scl_u256[nPublic+1:nVars])


        # Accumulate multiplication of S EC points and S scalar. Parallelization
        # can be accomplised by each thread performing a multiplication and storing
        # values in same input EC point
        # Second step is to add all ECC points
        pib1_eccf1 = self.findECPoints()

        d1 = d2 = d3 = 0
        # polH must be in extended format
        polH = self.calculateH(d1, d2, d3)
        
        if use_pycusnarks and self.accel:
          start = time.time()
          ZField.set_field(GrothProver.FieldIDX)
          d4_u256  = ZFieldElExt(-self.r_scl * self.s_scl).as_uint256()

          ZField.set_field(GrothProver.GroupIDX)
          self.pi_a_eccf1 = ec_jac2aff_h(self.pi_a_eccf1.reshape(-1),ZField.get_field())
          self.pi_b_eccf2 = ec2_jac2aff_h(self.pi_b_eccf2.reshape(-1),ZField.get_field())
          pib1_eccf1  = ec2_jac2aff_h(pib1_eccf1.reshape(-1), ZField.get_field())
          self.pi_c_eccf1 = ec_jac2aff_h(self.pi_c_eccf1.reshape(-1),ZField.get_field())

          one = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32)
          sorted_H_idx = sortu256_idx_h(polH)

          K = np.concatenate((polH[sorted_H_idx],[one],[self.s_scl_u256],[self.r_scl_u256],[d4_u256]))
          sorted_hExps = np.reshape(self.hExps_eccf1_u256[:2*nVars],(-1,2,NWORDS_256BIT))[sorted_H_idx]
          P = np.concatenate((np.reshape(sorted_hExps,(-1,NWORDS_256BIT)),self.pi_c_eccf1[:2], self.pi_a_eccf1[:2], pib1_eccf1[:2], self.vk_delta_1_eccf1_u256))
          ecbn128_samples = np.concatenate((K,P))

          self.pi_c_eccf1,t1 = ec_mad_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
          self.t_EC.append(t1)
          self.pi_c_eccf1 = ec_jac2aff_h(self.pi_c_eccf1.reshape(-1),ZField.get_field())
          self.public_signals = self.witness_scl_u256[1:self.vk_proof['nPublic']+1]
          end = time.time()
          self.t_GP.append(end - start)

        else :
          d4  = ZFieldElExt(-(self.r_scl * self.s_scl))
          ZField.set_field(GrothProver.GroupIDX)
          d5  = d4 * self.vk_delta_1_eccf1

          coeffH = polH.get_coeff()

          # Accumulate products of S ecc points and S scalars (same as at the beginning)
          n_coeff_h = len(coeffH)
          self.pi_c_eccf1 += np.sum(np.multiply(self.hExps_eccf1[:n_coeff_h ], coeffH[:n_coeff_h]))

        
          self.pi_c_eccf1  += (self.pi_a_eccf1 * self.s_scl) + (pib1_eccf1 * self.r_scl) + d5

          self.public_signals = self.witness_scl[1:self.vk_proof['nPublic']+1]

        return self.t_EC, self.t_P, self.t_GP
 

    def findECPoints(self):
        nVars = self.vk_proof['nVars']
        nPublic = self.vk_proof['nPublic']
        self.t_EC = []
        ZField.set_field(GrothProver.GroupIDX)

        if use_pycusnarks and self.accel:
          start_ec = time.time()
          #pi_a -> add 1 and r_u256 to scl, and alpha1 and delta1 to P 
          one = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32)
          sorted_scl = self.witness_scl_u256[:nVars][self.sorted_witness1_idx]
          sorted_A_ecc = np.reshape(self.A_eccf1_u256[:2*nVars],(-1,2,NWORDS_256BIT))[self.sorted_witness1_idx]
          K = np.concatenate((sorted_scl,[one], [self.r_scl_u256]))
          P = np.concatenate((np.reshape(sorted_A_ecc,(-1,NWORDS_256BIT)),self.vk_alfa_1_eccf1_u256, self.vk_delta_1_eccf1_u256))
          ecbn128_samples = np.concatenate((K,P))
          self.pi_a_eccf1,t1 = ec_mad_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
          self.t_EC.append(t1)

          # pi_b = pi_b + beta2 + delta2 * s
          K[-1] = self.s_scl_u256
          sorted_B2_ecc = np.reshape(self.B2_eccf2_u256[:4*nVars],(-1,4,NWORDS_256BIT))[self.sorted_witness1_idx]
          P = np.concatenate((np.reshape(sorted_B2_ecc,(-1,NWORDS_256BIT)),self.vk_beta_2_eccf2_u256, self.vk_delta_2_eccf2_u256))
          ec2bn128_samples = np.concatenate((K,P))
          self.pi_b_eccf2, t1 = ec_mad_cuda(self.ec2bn128, ec2bn128_samples, ZField.get_field(), ec2=True)
          self.t_EC.append(t1)
          

          # pib1 = pib1 + beta1 + delta1 * s
          sorted_B1_ecc = np.reshape(self.B1_eccf1_u256[:2*nVars],(-1,2,NWORDS_256BIT))[self.sorted_witness1_idx]
          P = np.concatenate((np.reshape(sorted_B1_ecc,(-1,NWORDS_256BIT)),self.vk_beta_1_eccf1_u256, self.vk_delta_1_eccf1_u256))
          ecbn128_samples = np.concatenate((K,P))
          pib1_eccf1, t1 = ec_mad_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
          self.t_EC.append(t1)

          # pi_c
          sorted_scl = self.witness_scl_u256[nPublic+1:nVars][self.sorted_witness2_idx]
          sorted_C_ecc = np.reshape(self.C_eccf1_u256[2*(nPublic+1):2*nVars],(-1,2,NWORDS_256BIT))[self.sorted_witness2_idx]
          ecbn128_samples = np.concatenate((sorted_scl, np.reshape(sorted_C_ecc,(-1,NWORDS_256BIT))))
          self.pi_c_eccf1,t1 = ec_mad_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
          self.t_EC.append(t1)
          end_ec = time.time()
          self.t_EC.append(end_ec - start_ec)

        else :
          self.pi_a_eccf1  = np.sum(np.multiply(self.A_eccf1[:nVars], self.witness_scl[:nVars]))
          self.pi_b_eccf2  = np.sum(np.multiply(self.B2_eccf2[:nVars], self.witness_scl[:nVars]))
          pib1_eccf1       = np.sum(np.multiply(self.B1_eccf1[:nVars], self.witness_scl[:nVars]))
          self.pi_c_eccf1  = np.sum(np.multiply(self.C_eccf1[nPublic+1:nVars], self.witness_scl[nPublic+1:nVars]))

          # pi_a = pi_a + alfa1 + delta1 * r
          self.pi_a_eccf1  += self.vk_alfa_1_eccf1
          self.pi_a_eccf1  += (self.vk_delta_1_eccf1 * self.r_scl)

          # pi_b = pi_b + beta2 + delta2 * s
          self.pi_b_eccf2  += self.vk_beta_2_eccf2
          self.pi_b_eccf2  += (self.vk_delta_2_eccf2 * self.s_scl)
  
          # pib1 = pib1 + beta1 + delta1 * s
          pib1_eccf1 += self.vk_beta_1_eccf1
          pib1_eccf1 += (self.vk_delta_1_eccf1 * self.s_scl)

        return pib1_eccf1


    def write_proof(self):
        if self.out_public_f.endswith('.json'):
           # Write public file
           ZField.set_field(GrothProver.FieldIDX)
           ps = [str(BigInt.from_uint256(el).as_long()) for el in self.public_signals]
           j = json.dumps(ps, indent=4)
           f = open(self.out_public_f, 'w')
           print(j,file=f)
           f.close()
        elif self.out_public_f.endswith('bin') :
           public_bin = np.concatenate((
                   np.asarray([self.public_signals.shape[0]], dtype=np.uint32),
                   np.reshape(self.public_signals,(-1))))
           writeU256CircuitFile_h(public_bin, self.out_public_f.encode("UTF-8"))

        if self.out_proof_f.endswith('.json'):
           ZField.set_field(GrothProver.GroupIDX)
           # write proof file
           P = ECC.from_uint256(self.pi_a_eccf1, in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)
           pi_a = [el.extend().as_str() for el in P]
   
           P = ECC.from_uint256(self.pi_c_eccf1, in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)
           pi_c = [el.extend().as_str() for el in P]
   
           P = ECC.from_uint256(np.reshape(self.pi_b_eccf2,(-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True, ec2=True)
           pi_b = [el.extend().as_str() for el in P]
           proof = {"pi_a" : pi_a, "pi_b" : pi_b, "pi_c" : pi_c, "protocol" : "groth"}
           j = json.dumps(proof, indent=4, sort_keys=True)
           f = open(self.out_proof_f, 'w')
           print(j, file=f)
           f.close()
        elif self.out_proof_f.endswith('.bin'):
           self.pi_a_eccf1 = from_montgomeryN_h(np.reshape(self.pi_a_eccf1,(-1)), GrothProver.GroupIDX)
           self.pi_b_eccf2 = from_montgomeryN_h(np.reshape(self.pi_b_eccf2,(-1)), GrothProver.GroupIDX)
           self.pi_c_eccf1 = from_montgomeryN_h(np.reshape(self.pi_c_eccf1,(-1)), GrothProver.GroupIDX)
           proof_bin = np.concatenate((
                    self.pi_a_eccf1, 
                    self.pi_b_eccf2,
                    self.pi_c_eccf1))
           writeU256CircuitFile_h(proof_bin, self.out_public_f.encode("UTF-8"))
               

    def calculateH(self, d1, d2, d3):
        ZField.set_field(GrothProver.FieldIDX)
        m = np.int32(self.vk_proof['domainSize'])
        nVars = self.nVars
        self.t_P = []

        if use_pycusnarks and self.accel:
          start_h = time.time()
          start = time.time()
          d2_u256 = ZFieldElExt(d2).reduce().as_uint256()
          d1d2 = ZFieldElExt(d1 * d2)
          _d3_d1d2 = ZFieldElExt(-d3 - d1d2.as_long())
          end = time.time()
          self.t_P.append(end-start)

          start = time.time()
          pidx = ZField.get_field()
          # Convert witness to montgomery in zpoly_maddm_h
          #polA_T, polB_T, polC_T are montgomery -> polsA_sps_u256, polsB_sps_u256, polsC_sps_u256 are montgomery
          reduce_coeff = 0
          polA_T = mpoly_eval_h(self.witness_scl_u256,self.polsA_sps_u256, reduce_coeff, m, nVars-1, pidx)
          polB_T = mpoly_eval_h(self.witness_scl_u256,self.polsB_sps_u256, reduce_coeff, m, nVars-1, pidx)
          polC_T = mpoly_eval_h(self.witness_scl_u256,self.polsC_sps_u256, reduce_coeff, m, nVars-1, pidx)
          end = time.time()
          self.t_P.append(end-start)
          end_h = time.time()
          t_h = end_h - start_h

          start_h = time.time()
          ifft_params = ntt_build_h(polA_T.shape[0]);
          # polC_S  is extended -> use extended scaler
          polC_S,t1 = zpoly_ifft_cuda(self.cuzpoly, polC_T[:nVars], ifft_params, ZField.get_field(), as_mont=0, roots=self.roots1M_rdc_u256)
          self.t_P.append(t1)
          # polA_S montgomery -> use montgomery scaler
          polA_S,t1 = zpoly_ifft_cuda(self.cuzpoly, polA_T[:nVars],ifft_params, ZField.get_field(), as_mont=1)
          self.t_P.append(t1)
          # polB_S montgomery  -> use montgomery scaler
          # TODO : return_val = 0, out_extra_len= out_len
          polB_S,t1 = zpoly_ifft_cuda(self.cuzpoly, polB_T[:nVars], ifft_params, ZField.get_field(), as_mont=1, return_val = 1, out_extra_len=0)
          self.t_P.append(t1)

          mul_params = ntt_build_h(polA_S.shape[0]*2)
          #polAB_S is extended -> use extended scaler
          # TODO : polB_S is stored in device mem already from previous operation. Do not return  value
          polAB_S,t1 = zpoly_mul_cuda(self.cuzpoly, polA_S,polB_S,mul_params, ZField.get_field(), roots=self.roots1M_rdc_u256, return_val=1, as_mont=0)
          self.t_P.append(t1)
          polAB_S = polAB_S[:zpoly_norm_h(polAB_S)]

          # polABC_S is extended
          # TODO : polAB_S is stored in device moem already from previous operatoin. Do not return value.
          # TODO : perform several sub operations per thread to improve efficiency
          polABC_S,t1 = zpoly_sub_cuda(self.cuzpoly, polAB_S, polC_S, ZField.get_field(), vectorA_len = 0, return_val=1)
          self.t_P.append(t1)
          polABC_S = polABC_S[:zpoly_norm_h(polABC_S)]

          # polABC_S, polH_S are extended
          polH_S = polABC_S[m:]

          #TODO : d1, d2 and d3 assumed to be zero
          start = time.time()
          polH_S_idx = zpoly_norm_h(polH_S)
          polH_S = polH_S[:zpoly_norm_h(polH_S)]
          end = time.time()
          self.t_P.append(end-start)
          end_h = time.time()
          self.t_P.append(end_h - start_h + t_h)
  
        else:
          # Init dense poly of degree m-1 (all zero)
          polA_T = ZPoly([ZFieldElExt(0) for i in xrange(m)])
          polB_T = ZPoly([ZFieldElExt(0) for i in xrange(m)])
          polC_T = ZPoly([ZFieldElExt(0) for i in xrange(m)])

          # interpretation of polsA is that there are up to S sparse polynomials, and each sparse poly
          # has C sparse coeffs
          # polA_T = polA_T + witness[s] * polsA[s]
          #
          #  for (let s=0; s<vk_proof.nVars; s++) {
          #     for (let c in vk_proof.polsA[s]) {
          #           polA_T[c] = F.add(polA_T[c], F.mul(witness[s], vk_proof.polsA[s][c]));
          #     }
          #
          # Ex:
          # s iterates from 0 to 4
          # polsA = [{'1': 1}, {'2' : 1}, {'0': 3924283749832748327}, {}]
          # polsA[0] = {'1' : 1}, polsA[1] = {'2':1}, polsA[2] = {'0',3243243284}, polsA[3]={}
          # c in polsA[0] : '1', c in polsA[1] : '2', c in polsA[2] : '0', c in polsA[3] : {}
          # polA_T[1] = polA_t[1] + (witness[0] * polsA[0]['1'])
          # polA_T[2] = polA_t[2] + (witness[1] * polsA[1]['2'])
          # polA_T[0] = polA_t[0] + (witness[2] * polsA[2]['0'])
  
  
          polA_T = np.sum( np.multiply([1] + self.witness_scl[:nVars], [polA_T] + self.polsA_sps[:nVars]))
          polB_T = np.sum( np.multiply([1] + self.witness_scl[:nVars], [polB_T] + self.polsB_sps[:nVars]))
          polC_T = np.sum( np.multiply([1] + self.witness_scl[:nVars], [polC_T] + self.polsC_sps[:nVars]))
  
          polA_T = polA_T.expand_to_degree(nVars-1)
          polB_T = polB_T.expand_to_degree(nVars-1)
          polC_T = polC_T.expand_to_degree(nVars-1)
  
          # in : poly deg nVars-1.  out : poly deg nVars - 1
          polA_S = ZPoly(polA_T)
          polA_S.intt()
          # in : poly deg nVars-1.  out : poly deg nVars - 1
          polB_S = ZPoly(polB_T)
          polB_S.intt()
 
          polC_S = ZPoly(polC_T)
          polC_S.intt()

          # in : 2xpoly deg nVars-1.  out : poly deg nVars - 1
          polAB_S = ZPoly(polA_S)
          polAB_S.poly_mul(ZPoly(polB_S))

          polABC_S = polAB_S - polC_S
    
          inv_pol = None
    
          polZ_S = ZPoly([ZFieldElExt(-1)] + [ZFieldElExt(0) for i in range(m-1)] + [ZFieldElExt(1)])

          #return polABC_S
          polH_S = polABC_S.poly_div_snarks(polZ_S.get_degree())

          """
          H_S_copy = ZPoly(polH_S)
          H_S_copy.poly_mul(polZ_S)

          if H_S_copy == polABC_S:
            print "OK"
          else:
            print "KO"
          """

          # add coefficients of the polynomial (d2*A + d1*B - d3) + d1*d2*Z 

          polH_S = polH_S + d2 * polA_S + d2 * polB_S
          polH_S = polH_S.expand_to_degree(m)

          polH_S.zcoeff[0] -= d3
  
          # Z = x^m -1
          d1d2 = d1 * d2
          polH_S.zcoeff[m] += d1d2
          polH_S.zcoeff[0] -= d1d2
    
          polH_S = polH_S.norm()
  
        return polH_S

if __name__ == "__main__":
    t = []
    witness_f=DEFAULT_WITNESS_LOC
    proving_key_f = DEFAULT_PROVING_KEY_LOC

    G = GrothProver(proving_key_f, inrand=True)
    for i in range(1):
      t1,t2,t3 = G.gen_proof(witness_f)
      t.append(np.concatenate((t1,t2,t3)))
      print(t[-1])
    print(t)
     
    G.write_proof()


"""  
   t:
     ECPoints-EC_MAD_CUDA, ECPoints-EC2_MAD_CUDA, ECPoints-EC_MAD_CUDA, ECPoints-EC_MAD_CUDA, ECPoints-All
     H-d, H-ZPOLY_MADDM_H, H-IFFT_C, H-IFFT_A, H-IFFT_B, H-MUL, H-AB_C, H-DIV, 0,0,0, H-NORM, H-All
     END- MAD_CUDA, END-all
         
"""
