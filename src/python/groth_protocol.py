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
// File name  : groth_protocol
//
// Date       : 27/01/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Zero Kowledge Groth protocol implementation
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

from zutils import ZUtils
from random import randint
from zfield import *
from ecc import *
from zpoly import *
from constants import *

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
DEFAULT_PROOF_LOC =  "../../data/proof.json"
DEFAULT_PUBLIC_LOC =  "../../data/public.json"
DATA_FOLDER = "../../data/"

class GrothSnarks(object):
    
    GroupIDX = 0
    FieldIDX = 1

    def __init__(self, curve='BN128', witness_f=DEFAULT_WITNESS_LOC, proving_key_f = DEFAULT_PROVING_KEY_LOC, proof_f = DEFAULT_PROOF_LOC, public_f = DEFAULT_PUBLIC_LOC, in_point_fmt=ZUtils.FEXT, work_point_fmt=ZUtils.FEXT, in_coor_fmt = ZUtils.JACOBIAN, work_coor_fmt= ZUtils.JACOBIAN, accel = True):

        self.accel = True
        data_init = self.read_data(witness_f, proving_key_f)

        self.proof_f = proof_f
        self.public_f = public_f
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data['curve_params'])
        ZPoly.init(GrothSnarks.FieldIDX)

        ZUtils.set_default_in_p_format(in_point_fmt)  # FEXT
        ZUtils.set_default_in_rep_format(in_coor_fmt) # AFFINE/PROJECTIVE/JACOBIAN

        if not data_init:
          #init class variables
          self.init_vars()

          if use_pycusnarks and self.accel:
             self.init_u256_vars(witness_f, proving_key_f)

        #TODO  : Remove once working in CUDA
        ZField.set_field(GrothSnarks.FieldIDX)
        ZField.find_roots(ZUtils.NROOTS)
        ZField.set_field(GrothSnarks.GroupIDX)

        if use_pycusnarks and self.accel:
          nroots = 1 << int(np.ceil(np.log2(self.vk_proof['nVars'])))
          nroots = 1024
          self.roots1_u256 = self.get_roots_u256(nroots)
          #TODO Inv roots = [roots[0], roots[-1], root[-2],...]
          #self.inv_roots
          self.roots2_u256 = self.get_roots_u256(nroots*2)

          #TODO Check this
          #store inv x^(n-1)-1 where n =vk_proof['domain_bits']
          # it seems the answer is always x^(n-1)+1. If this is the case, then it is easy
          m = self.vk_proof['domainSize']
          self.inv_poly_u256 = self.get_invpoly_u256(m)

        """
          TODO : pending capability to switch from FEXt to REDC and from/to affine/projective/affine
            right now, everything is ext and jacobian. Also witness_scl needs to be xformed

        """

    def read_data(self, witness_f, proving_key_f):
       ## Open and parse witness data
       witness_fnpz = witness_f[:-4] + 'npz'
       proving_key_fnpz = proving_key_f[:-4] + 'npz'
       data_init = False
      
       if use_pycusnarks and os.path.isfile(witness_fnpz)  and os.path.isfile(proving_key_fnpz) and self.accel:
          npzfile = np.load(witness_fnpz)
          self.witness_scl_u256 = npzfile['witness_u256']
       
       elif os.path.isfile(witness_f):
           f = open(witness_f,'r')
           self.witness_scl = [long(c) for c in ast.literal_eval(json.dumps(json.load(f)))]
           f.close()
       else:
          print "File doesn't exist"
          assert False
       
       if use_pycusnarks and os.path.isfile(proving_key_fnpz) and os.path.isfile(witness_fnpz) and self.accel:
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
           self.vk_proof = GrothSnarks.json_to_dict(tmp_data)
           f.close()
       else:
          print "File doesn't exist"
          assert False


       return data_init

    def init_u256_vars(self, witness_f, proving_key_f):
       self.witness_scl_u256 = \
          np.reshape(np.asarray([el.as_uint256() for el in self.witness_scl], dtype=np.uint32),(-1,NWORDS_256BIT))
       self.vk_alfa_1_eccf1_u256 = ECC.as_uint256(self.vk_alfa_1_eccf1,remove_last=True)
       self.vk_beta_1_eccf1_u256 = ECC.as_uint256(self.vk_beta_1_eccf1,remove_last=True)
       self.vk_delta_1_eccf1_u256= ECC.as_uint256(self.vk_delta_1_eccf1,remove_last= True)
       self.vk_beta_2_eccf2_u256 = ECC.as_uint256(self.vk_beta_2_eccf2, remove_last = True)
       self.vk_delta_2_eccf2_u256 = ECC.as_uint256(self.vk_delta_2_eccf2, remove_last = True)

       self.A_eccf1_u256 = ECC.as_uint256(self.A_eccf1, remove_last = True)
       self.B1_eccf1_u256 = ECC.as_uint256(self.B1_eccf1, remove_last = True)
       self.B2_eccf2_u256 = ECC.as_uint256(self.B2_eccf2, remove_last = True)
       self.C_eccf1_u256  = ECC.as_uint256(self.C_eccf1, remove_last=True)
       self.hExps_eccf1_u256 = ECC.as_uint256(self.hExps_eccf1, remove_last=True)

       polsA_l = []
       polsA_p = []
       for pol in self.polsA_sps:
         l,p = pol.as_uint256() 
         polsA_l.append(l)
         polsA_p.append(p)
       self.polsA_sps_u256 = np.asarray(np.concatenate((np.asarray([len(polsA_l)]),
                                    np.concatenate((np.cumsum(polsA_l),np.concatenate(polsA_p))))),dtype=np.uint32)

       polsB_l = []
       polsB_p = []
       for pol in self.polsB_sps:
         l,p = pol.as_uint256() 
         polsB_l.append(l)
         polsB_p.append(p)
       self.polsB_sps_u256 = np.asarray(np.concatenate((np.asarray([len(polsB_l)]),
                                    np.concatenate((np.cumsum(polsB_l),np.concatenate(polsB_p))))),dtype=np.uint32)

       polsC_l = []
       polsC_p = []
       for pol in self.polsC_sps:
         l,p = pol.as_uint256() 
         polsC_l.append(l)
         polsC_p.append(p)
       self.polsC_sps_u256 = np.asarray(np.concatenate((np.asarray([len(polsC_l)]),
                                    np.concatenate((np.cumsum(polsC_l),np.concatenate(polsC_p))))),dtype=np.uint32)

       
       witness_fnpz = witness_f[:-4] + 'npz'
       proving_key_fnpz = proving_key_f[:-4] + 'npz'

       np.savez_compressed(witness_fnpz, witness_u256=self.witness_scl_u256)
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

       ZField.set_field(GrothSnarks.FieldIDX)
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
 
    def get_roots_u256(self, nroots):
       fidx = ZField.get_field()

       ZField.set_field(GrothSnarks.FieldIDX)
       root_fnpz = DATA_FOLDER + "root_"+str(nroots)+".npz"
       if not os.path.isfile(root_fnpz):
         proot = ZField.find_primitive_root(nroots).reduce().as_uint256()
         roots = find_roots_h( proot, nroots, ZField.get_field())
         np.savez_compressed(root_fnpz, root_data=roots)
       else:  
          npzfile = np.load(root_fnpz)
          roots = npzfile['root_data']

       ZField.set_field(fidx)

       return roots

    def init_vars(self):
        # Init witness to Field El.
        # TODO :  I am assuming that all field el are FielElExt (witness_scl, polsA_sps, polsB_sps, polsC_sps, alfa1...
        # Witness is initialized a BitInt as it will operate on different fields
        self.witness_scl = [BigInt(el) for el in self.witness_scl]

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

        self.A_eccf1     = map(ECC_F1,self.vk_proof['A'])
        self.B1_eccf1    = map(ECC_F1,self.vk_proof['B1'])
        self.B2_eccf2    = map(ECC_F2, self.vk_proof['B2'])
        self.C_eccf1     = map(ECC_F1,self.vk_proof['C'])
        self.hExps_eccf1 = map(ECC_F1,self.vk_proof['hExps'])


        ZField.set_field(GrothSnarks.FieldIDX)

        ZField.find_roots(ZUtils.NROOTS)
        # TODO : This representation may not be optimum. I only have good representation of sparse polynomial,
        #  but not of array of sparse poly (it is also sparse). I should encode it as a dictionary as wekk
        self.polsA_sps = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in self.vk_proof['polsA']]
        self.polsB_sps = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in self.vk_proof['polsB']]
        self.polsC_sps = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in self.vk_proof['polsC']]

        #TODO
        """"
        if self.in_point_fmt == FRDC:
            r_scl = r_scl.reduce()
            s_scl = s_scl.reduce()
        """
        ZField.set_field(GrothSnarks.GroupIDX)


    @classmethod
    def json_to_dict(cls, data):
       data = {str(k) : data[k] for k in data.keys()}
       for k, v in data.items():
           if type(data[k]) is list:
               GrothSnarks.json_to_list(data[k])
           elif type(data[k]) is dict:
               GrothSnarks.json_to_dict(data[k])
           elif GrothSnarks.is_long(v):
               data[k] = long(v)

       return data

    @classmethod
    def json_to_list(cls,data):
        for idx,el in enumerate(data):
            if type(el) is list:
                GrothSnarks.json_to_list(el)
            elif type(el) is dict:
                data[idx] = GrothSnarks.json_to_dict(el)
            elif type(el) is unicode or type(el) is str:
                data[idx] = long(el)
        return

    @classmethod
    def is_long(cls, input):
        try:
            num = long(input)
        except ValueError:
            return False
        return True

    def gen_proof(self, piter):
        """
          public_signals, pi_a_eccf1, pi_b_eccf2, pi_c_eccf1 
        """
        self.proof_iter = piter
        ZField.set_field(GrothSnarks.FieldIDX)
        # Init r and s scalars
        self.r_scl = BigInt(randint(1,ZField.get_extended_p().as_long()-1))
        self.s_scl = BigInt(randint(1,ZField.get_extended_p().as_long()-1))
        if use_pycusnarks and self.accel:
           self.r_scl_u256 = self.r_scl.as_uint256()
           self.s_scl_u256 = self.s_scl.as_uint256()
        ZField.set_field(GrothSnarks.GroupIDX)


        # init Pi B1 ECC point
        #pi_a, pi_b, pi_c, pib1 F (mod q)
        #pib1_eccf1 = ECC_F1()

        #TODO
        self.witness_scl_u256=np.tile(self.witness_scl_u256,(2*1024*1024//len(self.witness_scl_u256)+20,1))
        self.A_eccf1_u256 = np.tile(self.A_eccf1_u256,(2*1024*1024//len(self.A_eccf1_u256)+20,1))
        self.B2_eccf2_u256 = np.tile(self.B2_eccf2_u256,(4*1024*1024//len(self.B2_eccf2_u256)+20,1))
        self.B1_eccf1_u256 = np.tile(self.B1_eccf1_u256,(2*1024*1024//len(self.B1_eccf1_u256)+20,1))
        self.C_eccf1_u256 = np.tile(self.C_eccf1_u256,(2*1024*1024//len(self.C_eccf1_u256)+20,1))
        self.oldnVars = self.vk_proof['nVars']
        self.vk_proof['nVars'] = 1024*1024
        self.roots1_u256 = self.get_roots_u256(1024*1024)

        # Accumulate multiplication of S EC points and S scalar. Parallelization
        # can be accomplised by each thread performing a multiplication and storing
        # values in same input EC point
        # Second step is to add all ECC points
        pib1_eccf1 = self.findECPoints()


        ZField.set_field(GrothSnarks.FieldIDX)
        d1 = d2 = d3 = 0
        polH = self.calculateH(d1, d2, d3)
        #TODO
        
        return self.t_EC, self.t_P
        coeffH = polH.get_coeff()

        d4  = ZFieldElExt(-(self.r_scl * self.s_scl))
        ZField.set_field(GrothSnarks.GroupIDX)
        d5  = d4 * self.vk_delta_1_eccf1

        # Accumulate products of S ecc points and S scalars (same as at the beginning)
        n_coeff_h = len(coeffH)
        self.pi_c_eccf1 += np.sum(np.multiply(self.hExps_eccf1[:n_coeff_h ], coeffH[:n_coeff_h]))

        
        self.pi_c_eccf1  += (self.pi_a_eccf1 * self.s_scl) + (pib1_eccf1 * self.r_scl) + d5

        self.public_signals = self.witness_scl[1:self.vk_proof['nPublic']+1]


    def findECPoints(self):
        nVars = self.vk_proof['nVars']
        nPublic = self.vk_proof['nPublic']
        self.t_EC = []

        if use_pycusnarks and self.accel:
          # add 1 and r_u256 to scl, and alpha1 and delta1 to P
          one = np.asarray([1,0,0,0,0,0,0,0],dtype=np.uint32)
          K = np.concatenate((self.witness_scl_u256[:nVars],[one], [self.r_scl_u256]))
          P = np.concatenate((self.A_eccf1_u256[:2*nVars],self.vk_alfa_1_eccf1_u256, self.vk_delta_1_eccf1_u256))
          ecbn128_samples = np.concatenate((K,P))
          if self.proof_iter == 0:
             ecbn128 = ECBN128(len(ecbn128_samples)/3, seed=1)
          self.pi_a_eccf1,t1 = ec_mad_cuda(ecbn128, ecbn128_samples, ZField.get_field())
          self.t_EC.append(t1)

          # pi_b = pi_b + beta2 + delta2 * s
          K = np.concatenate((self.witness_scl_u256[:nVars],[one], [self.s_scl_u256]))
          P = np.concatenate((self.B2_eccf2_u256[:4*nVars],self.vk_beta_2_eccf2_u256, self.vk_delta_2_eccf2_u256))
          ec2bn128_samples = np.concatenate((K,P))
          if self.proof_iter == 0:
             ec2bn128 = EC2BN128(len(ec2bn128_samples)/5, seed=1)
          self.pi_b_eccf2, t1 = ec2_mad_cuda(ec2bn128, ec2bn128_samples[:1024*1024*5], ZField.get_field())
          self.t_EC.append(t1)
          

          # pib1 = pib1 + beta1 + delta1 * s
          P = np.concatenate((self.B1_eccf1_u256[:2*nVars],self.vk_beta_1_eccf1_u256, self.vk_delta_1_eccf1_u256))
          #ecbn128_samples = np.concatenate((self.witness_scl_u256[:nVars], self.B1_eccf1_u256[:2*nVars]))
          ecbn128_samples = np.concatenate((K,P))
          pib1_eccf1, t1 = ec_mad_cuda(ecbn128, ecbn128_samples, ZField.get_field())
          self.t_EC.append(t1)

          ecbn128_samples = np.concatenate((self.witness_scl_u256[nPublic+1:nVars], self.C_eccf1_u256[2*(nPublic+1):2*nVars]))
          self.pi_c_eccf1,t1 = ec_mad_cuda(ecbn128, ecbn128_samples, ZField.get_field())
          self.t_EC.append(t1)

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

    def write_json(self):
        # Write public file
        ps = [str(el.as_long()) for el in self.public_signals]
        j = json.dumps(ps, indent=4).encode('utf8')
        f = open(self.public_f, 'w')
        print >> f, j
        f.close()

        # write proof file
        pi_a = [str(el) for el in self.pi_a_eccf1.to_affine().as_list()]
        pi_c = [str(el) for el in self.pi_c_eccf1.to_affine().as_list()]
        pi_b = []
        pi_b_els = [el for el in self.pi_b_eccf2.to_affine().as_list()]
        for i in range(len(pi_b_els)):
           pi_b.append([str(el) for el in pi_b_els[i]])
        proof = {"pi_a" : pi_a, "pi_b" : pi_b, "pi_c" : pi_c, "protocol" : "groth"}
        j = json.dumps(proof, indent=4).encode('utf8')
        f = open(self.proof_f, 'w')
        print >> f, j
        f.close()

    def calculateH(self, d1, d2, d3):
        #d1 = PolF.F.zero, d2 = PolF.F.zero, d3 = PolF.F.zero);

        m = self.vk_proof['domainSize']
        #nVars = self.vk_proof['nVars']
        nVars = self.oldnVars
        self.t_P = []

        if use_pycusnarks and self.accel:
          pidx = ZField.get_field()
          s = np.copy(self.polsA_sps_u256)
          polA_T = zpoly_maddm_h(self.witness_scl_u256,self.polsA_sps_u256, nVars, nVars-1, pidx)
          polB_T = zpoly_maddm_h(self.witness_scl_u256,self.polsB_sps_u256, nVars, nVars-1, pidx)
          polC_T = zpoly_maddm_h(self.witness_scl_u256,self.polsC_sps_u256, nVars, nVars-1, pidx)

          #TODO
          polA_T = np.tile(polA_T,(1024*1040/len(polA_T),1))
          polB_T = np.tile(polB_T,(1024*1040/len(polB_T),1))
          polC_T = np.tile(polC_T,(1024*1040/len(polC_T),1))
          nVars = self.vk_proof['nVars']

          #TODO -> get inverse roots
          if self.proof_iter == 0:
            cuzpoly = ZCUPoly(2*len(polA_T), seed=560)
            cuu256 = U256(3*len(polA_T), seed=560)

          polA_S,t1 = zpoly_ifft_cuda(cuzpoly, polA_T[:nVars], self.roots1_u256, ZField.get_field())
          self.t_P.append(t1)
          polB_S,t1 = zpoly_ifft_cuda(cuzpoly, polB_T[:nVars], self.roots1_u256, ZField.get_field())
          self.t_P.append(t1)
          polC_S,t1 = zpoly_ifft_cuda(cuzpoly, polC_T[:nVars], self.roots1_u256, ZField.get_field())
          self.t_P.append(t1)

          polAB_S,t1 = zpoly_mul_cuda(cuzpoly, cuu256,polA_S[:nVars],polB_S[:nVars],self.roots1_u256, ZField.get_field())
          self.t_P.append(t1)

          polABC_S,t1 = zpoly_subm_cuda(cuu256, polAB_S, polC_S, ZField.get_field())
          self.t_P.append(t1)

          #polA_S = ZPoly.from_uint256(polA_S)
          #polB_S = ZPoly.from_uint256(polB_S)
          #polC_S = ZPoly.from_uint256(polC_S)
          return polABC_S
          polABC_S = ZPoly.from_uint256(polABC_S)
          nroots = 2 << int(np.ceil(np.log2(self.vk_proof['nVars'])))
          polABC_S = polABC_S.expand_to_degree(nroots-1)
          
          inv_pol = ZPoly.from_uint256(self.inv_poly_u256)
          #inv_pol = None
        

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
    
        #TODO Review this is the correct degree of poly
        polZ_S = ZPoly([ZFieldElExt(-1)] + [ZFieldElExt(0) for i in range(m-1)] + [ZFieldElExt(1)])

        #TODO
        return polABC_S
        polH_S = polABC_S.poly_div(polZ_S, invpol = inv_pol)

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

def ec2_mad_cuda(pysnark, vector, fidx):
     kernel_params={}
     kernel_config={}
     nsamples = len(vector)/5

     kernel_params['stride'] = [ECP2_JAC_INDIMS+U256_NDIMS, ECP2_JAC_OUTDIMS, ECP2_JAC_OUTDIMS]
     #kernel_config['blockD'] = [256,32]
     kernel_config['blockD'] = [256,128,32]
     kernel_params['premul'] = [1,0, 0]
     kernel_params['premod'] = [0,0, 0]
     kernel_params['midx'] = [fidx, fidx, fidx]
     kernel_config['smemS'] = [kernel_config['blockD'][0]/32 * NWORDS_256BIT * ECP2_JAC_OUTDIMS * 4, \
                               kernel_config['blockD'][1]/32 * NWORDS_256BIT * ECP2_JAC_OUTDIMS * 4, \
                               kernel_config['blockD'][2]/32 * NWORDS_256BIT * ECP2_JAC_OUTDIMS * 4]
     kernel_config['kernel_idx'] = [CB_EC2_JAC_MAD_SHFL, CB_EC2_JAC_MAD_SHFL, CB_EC2_JAC_MAD_SHFL]
     out_len1 = ECP2_JAC_OUTDIMS * ((nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP2_JAC_OUTDIMS) -1) /
                                   (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP2_JAC_OUTDIMS))
     out_len2 = ECP2_JAC_OUTDIMS * ((out_len1 + (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP2_JAC_OUTDIMS) -1) /
                                   (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP2_JAC_OUTDIMS))
     kernel_params['in_length'] = [nsamples * (ECP2_JAC_INDIMS+U256_NDIMS), out_len1, out_len2]
     kernel_params['out_length'] = 1 * ECP2_JAC_OUTDIMS
     kernel_params['padding_idx'] = [0,0, 0]
     kernel_config['gridD'] = [0,1, 1]
     min_length = [ECP2_JAC_OUTDIMS * \
             (kernel_config['blockD'][idx] * kernel_params['stride'][idx]/ECP2_JAC_OUTDIMS) for idx in range(len(kernel_params['stride']))]

     v_mad = np.copy(vector)
     for bidx, l in enumerate(kernel_params['in_length']):
        if l < min_length[bidx]:
           if bidx == 0:
              zeros = np.zeros((min_length[bidx] - kernel_params['in_length'][bidx],NWORDS_256BIT), dtype=np.uint32)
              v_mad = np.concatenate((vector,zeros))
              kernel_params['in_length'][bidx] = min_length[bidx]
           else:
              kernel_params['in_length'][bidx] = min_length[bidx]
              kernel_params['padding_idx'][bidx] = l/ECP2_JAC_OUTDIMS

     result,t = pysnark.kernelLaunch(v_mad, kernel_config, kernel_params,3 )
     
     return result,t
     
def ec_mad_cuda(pysnark, vector, fidx):
   kernel_params={}
   kernel_config={}
   nsamples = len(vector)/3

   kernel_params['stride'] = [ECP_JAC_OUTDIMS, ECP_JAC_OUTDIMS, ECP_JAC_OUTDIMS]
   #kernel_config['blockD'] = [256,32]
   kernel_config['blockD'] = [256,128,32]
   kernel_params['premul'] = [1,0,0]
   kernel_params['premod'] = [0,0,0]
   kernel_params['midx'] = [fidx, fidx, fidx]
   kernel_config['smemS'] = [kernel_config['blockD'][0]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4, \
                             kernel_config['blockD'][1]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4, \
                             kernel_config['blockD'][2]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4]
   kernel_config['kernel_idx'] = [CB_EC_JAC_MAD_SHFL, CB_EC_JAC_MAD_SHFL, CB_EC_JAC_MAD_SHFL]
   out_len1 = ECP_JAC_OUTDIMS * ((nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS) -1) /
                                 (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS))
   out_len2 = ECP_JAC_OUTDIMS * ((out_len1 + (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP_JAC_OUTDIMS) -1) /
                                 (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP_JAC_OUTDIMS))
   kernel_params['in_length'] = [nsamples * (ECP_JAC_INDIMS+U256_NDIMS), out_len1, out_len2]
   kernel_params['out_length'] = 1 * ECP_JAC_OUTDIMS
   kernel_params['padding_idx'] = [0,0,0]
   kernel_config['gridD'] = [0,1,1]
   min_length = [ECP_JAC_OUTDIMS * \
           (kernel_config['blockD'][idx] * kernel_params['stride'][idx]/ECP_JAC_OUTDIMS) for idx in range(len(kernel_params['stride']))]

   v_mad = np.copy(vector)
   for bidx, l in enumerate(kernel_params['in_length']):
      if l < min_length[bidx]:
         if bidx == 0:
            zeros = np.zeros((min_length[bidx] - kernel_params['in_length'][bidx],NWORDS_256BIT), dtype=np.uint32)
            v_mad = np.concatenate((vector,zeros))
            kernel_params['in_length'][bidx] = min_length[bidx]
         else:
            kernel_params['in_length'][bidx] = min_length[bidx]
            kernel_params['padding_idx'][bidx] = l/ECP_JAC_OUTDIMS

   result,t = pysnark.kernelLaunch(v_mad, kernel_config, kernel_params,3 )
   
   return result, t

def zpoly_fft_cuda(pysnark, vector, roots, fidx ):
        nsamples = len(vector)

        n_cols = 10
        n_rows = 10
        kernel_config={}
        kernel_params={}

        # Test FFT kernel:
        kernel_params['in_length'] = [2*nsamples,nsamples, nsamples, nsamples]
        kernel_params['out_length'] = nsamples
        kernel_params['stride'] = [2,1,1,1]
        kernel_params['premod'] = [0,0,0,0]
        kernel_params['midx'] = [fidx, fidx, fidx, fidx]
        kernel_params['fft_Nx'] = [5,5,5,5]
        kernel_params['fft_Ny'] = [5,5,5,5]
        kernel_params['N_fftx'] = [n_cols, n_cols, n_cols, n_cols]
        kernel_params['N_ffty'] = [n_rows, n_rows, n_rows, n_rows]
        kernel_params['forward'] = [1,1,1,1]

        kernel_config['smemS'] = [0,0,0,0]
        kernel_config['blockD'] = [256,256,256,256]
        kernel_config['gridD'] = [0, (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][1] + nsamples-1)/kernel_config['blockD'][1], \
                                     (kernel_config['blockD'][2] + nsamples-1)/kernel_config['blockD'][2]]
        kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]
        zpoly_vector = np.concatenate((vector, roots))
        result,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,4)

        return result, t


def zpoly_ifft_cuda(pysnark, vector, inv_roots, fidx ):
        nsamples = len(vector)

        kernel_config={}
        kernel_params={}
        n_cols = 10
        n_rows = 10

        # Test FFT kernel:
        kernel_params['in_length'] = [2*nsamples,nsamples, nsamples, nsamples]
        kernel_params['out_length'] = nsamples
        kernel_params['stride'] = [2,1,1,1]
        kernel_params['premod'] = [0,0,0,0]
        kernel_params['midx'] = [fidx, fidx, fidx, fidx]
        kernel_params['fft_Nx'] = [5,5, 5, 5]
        kernel_params['fft_Ny'] = [5,5, 5,5]
        kernel_params['forward'] = [0,0,0,0]

        kernel_config['smemS'] = [0,0,0,0]
        kernel_config['blockD'] = [256,256,256,256]
        kernel_config['gridD'] = [0, (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][1] + nsamples-1)/kernel_config['blockD'][1], \
                                     (kernel_config['blockD'][2] + nsamples-1)/kernel_config['blockD'][2]]
        kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]
        zpoly_vector = np.concatenate((vector, inv_roots))
        result,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,4)

        return result,t

def zpoly_mul_cuda(pysnark, cuu256, vectorA, vectorB, roots, fidx):
    t = 0
    rA,t1 = zpoly_fft_cuda(pysnark, vectorA, roots, fidx)
    t+=t1
    rB,t1 = zpoly_fft_cuda(pysnark, vectorB, roots, fidx)
    t+=t1
    #TODO : implement coeff mult kernel
    #cuu256 = U256(2*len(rA), seed=560)
    rC,t1 = zpoly_coeffmul_cuda(cuu256, rA, rB, fidx)
    t+=t1
    rD,t1 = zpoly_ifft_cuda(pysnark, rC, roots, fidx)
    t+=t1
    return rD, t

def zpoly_subm_cuda(pysnark, vectorA, vectorB, fidx):  
     #TODO revie
     vector =np.concatenate((vectorA, vectorB))
     nsamples = len(vector)
     kernel_config={}
     kernel_params={}

     kernel_params['premod'] = [0]
     kernel_params['in_length'] = [nsamples]
     kernel_params['out_length'] = nsamples/2
     kernel_params['stride'] = [2]
     kernel_params['midx'] = [fidx]
     kernel_config['smemS'] = [0]
     kernel_config['blockD'] = [U256_BLOCK_DIM]
     kernel_config['kernel_idx'] = [CB_U256_SUBM]
     result,t = pysnark.kernelLaunch(vector, kernel_config, kernel_params )

     return vector,t

def zpoly_coeffmul_cuda(pysnark, vectorA, vectorB, fidx):  
     #TODO revie
     vector =np.concatenate((vectorA, vectorB))
     nsamples = len(vector)
     kernel_config={}
     kernel_params={}

     kernel_params['in_length'] = [nsamples]
     kernel_params['out_length'] = nsamples/2
     kernel_params['stride'] = [2]
     kernel_params['midx'] = [fidx]
     kernel_params['premod'] = [0]
     kernel_config['smemS'] = [0]
     kernel_config['blockD'] = [U256_BLOCK_DIM]
     kernel_config['kernel_idx'] = [CB_U256_MULM]
     result,t= pysnark.kernelLaunch(vector, kernel_config, kernel_params )

     return result,t



if __name__ == "__main__":
    t = []
    G = GrothSnarks()
    for i in range(2):
      t1,t2 = G.gen_proof(i)
      t.append(np.concatenate((t1,t2)))
    print t
     
    #G.write_json()


