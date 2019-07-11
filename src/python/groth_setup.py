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
// File name  : groth_setup
//
// Date       : 13/05/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Zero Kowledge Groth setup implementation
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
from cuda_wrapper import *
from pysnarks_utils import *
from multiproc  import *
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

ROOTS_1M_filename_npz = '../../data/zpoly_data_1M.npz'
ROOTS_1M_filename_bin = '../../data/zpoly_roots_1M.bin'

class GrothSetup(object):
    

    def __init__(self, curve='BN128', in_circuit_f=None, out_circuit_f=None, out_circuit_format=FMT_MONT,
                 out_pk_f=None, out_vk_f=None, out_k_binformat=FMT_MONT, out_k_ecformat=EC_T_AFFINE, test_f=None,
                 benchmark_f=None, seed=None):
 
        if seed is not None:
          random.seed(seed) 

        self.worker = [mp.Pool() for p in range(mp.cpu_count()-1)]
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data)
        ZPoly.init(MOD_FIELD)

        ZField.set_field(MOD_GROUP)
        self.group_p        = ZField.get_extended_p().as_uint256()

        ZField.set_field(MOD_FIELD)
        self.field_p        = ZField.get_extended_p().as_uint256()
        self.Rbitlen        = np.asarray(ZField.get_reduction_data()['Rbitlen'],dtype=np.uint32)
   
        self.protocol     = PROTOCOL_T_GROTH  # Groth
        self.nWords       = None
        self.nPubInputs   = None
        self.nOutputs     = None
        self.nVars        = None
        self.nConstraints = None
        self.cirformat       = None
        self.R1CSA_nWords = None
        self.R1CSB_nWords = None
        self.R1CSC_nWords = None
        self.R1CSA        = None
        self.R1CSB        = None
        self.R1CSC        = None
        self.header       = None

        self.nPublic      = None
        self.domainBits   = None
        self.domainSize   = None
        self.A  = None
        self.B1 = None 
        self.B2 = None 
        self.C  = None
        self.vk_alfa_1  =  None
        self.vk_beta_1  =  None
        self.vk_delta_1 =  None
        self.vk_beta_2  =  None
        self.vk_delta_2 =  None
        self.polsA      =  None
        self.polsB      =  None
        self.polsC      =  None
        self.hExps      =  None 
        self.IC         =  None

        self.ecbn128    = None 
        self.ec2bn128   = None 

        self.out_pk_f   = out_pk_f
        self.out_k_binformat = out_k_binformat
        self.out_k_ecformat = out_k_ecformat

        self.test_f = test_f
        self.toxic = {}
 
        self.in_circuit_f = in_circuit_f
        self.out_circuit_f = out_circuit_f
        self.out_circuit_format = out_circuit_format
        self.in_circuit_format = FMT_EXT

        if self.in_circuit_f is not None:
           self.circuitRead()
        else:
           print("Input circuit is required\n")

    def circuitRead(self):
        # cir Json to u256
        if self.in_circuit_f.endswith('.json'):
           cir_u256 = self._cirjson_to_u256()

           #u256 to bin
           if self.out_circuit_f is not None:
              self._ciru256_to_bin(cir_u256)

        elif self.in_circuit_f.endswith('.bin'):
             cir_u256 = self._cirbin_to_u256()

        self._ciru256_to_vars(cir_u256)

    def launch_snarkjs(self):
        # snarkjs setup is launched with circuit.json, format extended. Convert input file if necessary
        if self.in_circuit_f.endswith('.json') and self.in_circuit_format == FMT_EXT
           circuit_file = self.in_circuit_f
        else :
           print("To launch snarkjs, input circuit needs to be a json file and format cannot be Montgomery")
           return
        
        tmp_pk_f = self.out_pk_f
        tmp_vk_f = self.out_vk_f   
        call(["snarkjs", "setup", "-c", circuit_file, "--pk", tmp_pk_f, "--vk", tmp_vk_f, "--protocol", "groth"])

        return tmp_pk_f, tmp_vk_f
   
    def setup(self):
        ZField.set_field(MOD_FIELD)
        self.domainBits =  np.uint32(math.ceil(math.log(self.nConstraints+ 
                                           self.nPubInputs + 
                                           self.nOutputs,2)))


        self.nPublic    = self.nPubInputs + self.nOutputs
        self.domainSize = 1 << self.domainBits

        prime = ZField.get_extended_p()

        self.toxic['t'] = ZFieldElExt(randint(1,prime.as_long()-1))

        self._calculatePoly()
        self._calculateEncryptedValuesAtT()

        self.write_pk()
        self.write_vk()

        if self.test_f is not None:
          self.write_toxic()
          tmp_pk_f, tmp_vk_f = self.launch_snarkjs("setup")
          self.compare_pk(tmp_pk_f)
          self.compare_vk(tmp_vk_f)

        return 


    def _calculatePoly(self):
        self._computeHeader()

        """ 
        r1 = mp.worker[0].apply_async(self._r1cs_to_mpoly, args=(self.R1CSA, self.header, 1))

        r2 = mp.worker[1].apply_async(self._r1cs_to_mpoly, args=(self.R1CSB, self.header, 0))

        r3 = mp.worker[2].apply_async(self._r1cs_to_mpoly, args=(self.R1CSC, self.header, 0))

        self.polsA = r1.get()
        self.R1CSA = None
        self.polsB = r2.get()
        self.r1csb = none
        self.polsC = r3.get()
        self.R1CSC = None
        return
        """
     
        self.polsA = self._r1cs_to_mpoly(self.R1CSA, self.header,1)
        self.R1CSA = None
        self.polsB = self._r1cs_to_mpoly(self.R1CSB, self.header,0)
        self.R1CSB = None
        self.polsC = self._r1cs_to_mpoly(self.R1CSC, self.header,0)
        self.R1CSC = None


    def _r1cs_to_mpoly(self, r1cs, header,extend):
        to_mont = 0
        pidx = ZField.get_field()
        if self.cirformat == ZUtils.FEXT:
           to_mont = 1

        poly_len = r1cs_to_mpoly_len_h(r1cs, header, extend)
        pols = r1cs_to_mpoly_h(poly_len, r1cs, header, to_mont, pidx, extend)
        
        return pols

    def _evalLagrangePoly(self, bits):
       """
        m : int
        t : ZFieldElRedc
       """
       m = 1 << bits
       t_rdc = self.toxic['t'].reduce()
       tm = (t_rdc ** int(m))
       u_u256 = np.zeros((m,NWORDS_256BIT),dtype=np.uint32)
       trdc_u256 = t_rdc.as_uint256()
      
       #load roots
       if os.path.exists(ROOTS_1M_filename_bin):
           roots_rdc_u256 = readU256DataFile_h(ROOTS_1M_filename_bin.encode("UTF-8"), 1<<20, 1<<bits)
       else :
           roots_rdc_u256 = field_roots_compute_h(nbits)

       z = tm.extend() - 1
       z_rdc = z.reduce()
       if tm == ZFieldElExt(1).reduce():
         for i in xrange(m): 
           #TODO : roots[0] is always 1. Does this make any sense? check javascript version
           if roots_rdc_u256[0] == trdc_u256:
             u_u256[i] = ZFieldElExt(1).reduce().as_uint256()
             return z.as_uint256(), u_u256

       l_rdc = z_rdc * ZFieldElExt(int(m)).inv().reduce()
       lrdc_u256 = l_rdc.as_uint256()

       pidx = ZField.get_field()
       u_u256 = evalLagrangePoly_h(trdc_u256,lrdc_u256, roots_rdc_u256, pidx)

       return z.as_uint256(), u_u256
   
    def _calculateEncryptedValuesAtT(self):
      a_t_u256, b_t_u256, c_t_u256, z_t_u256 = self._calculateValuesAtT()

      prime = ZField.get_extended_p()
      curve_params = self.curve_data['curve_params']
      curve_params_g2 = self.curve_data['curve_params_g2']

      # Tocix k extended
      self.toxic['kalfa'] = ZFieldElExt(randint(1,prime.as_long()-1))
      self.toxic['kbeta'] = ZFieldElExt(randint(1,prime.as_long()-1))
      self.toxic['kdelta'] = ZFieldElExt(randint(1,prime.as_long()-1))
      self.toxic['kgamma'] = ZFieldElExt(randint(1,prime.as_long()-1))

      toxic_invDelta = self.toxic['kdelta'].inv()
      toxic_invGamma = self.toxic['kgamma'].inv()

      ZField.set_field(MOD_GROUP)
      Gx = ZFieldElExt(curve_params['Gx'])
      Gy = ZFieldElExt(curve_params['Gy'])
      G2x = Z2FieldEl([curve_params_g2['Gx1'], curve_params_g2['Gx2']])
      G2y = Z2FieldEl([curve_params_g2['Gy1'], curve_params_g2['Gy2']])

      G1 = ECCJacobian([Gx,Gy]).reduce()
      G2 = ECCJacobian([G2x, G2y]).reduce()

      # vk coeff MONT
      self.vk_alfa_1 = G1 * self.toxic['kalfa']
      self.vk_beta_1 = G1 * self.toxic['kbeta']
      self.vk_delta_1 = G1 * self.toxic['kdelta']

      self.vk_alfa_1 = ec_jac2aff_h(G1.as_uint256(self.vk_alfa_1).reshape(-1),ZField.get_field())
      self.vk_beta_1 = ec_jac2aff_h(G1.as_uint256(self.vk_beta_1).reshape(-1),ZField.get_field())
      self.vk_delta_1 = ec_jac2aff_h(G1.as_uint256(self.vk_delta_1).reshape(-1),ZField.get_field())
    
      self.vk_beta_2 = G2 * self.toxic['kbeta']
      self.vk_delta_2 = G2 * self.toxic['kdelta']
    
      self.vk_beta_2 = ec2_jac2aff_h(G2.as_uint256(self.vk_beta_2).reshape(-1),ZField.get_field())
      self.vk_delta_2 = ec2_jac2aff_h(G2.as_uint256(self.vk_delta_2).reshape(-1),ZField.get_field())

      self.ecbn128    =  ECBN128(self.domainSize+3,seed=1)
      self.ec2bn128    = EC2BN128(self.nVars+1,seed=1)
   
      # a_t, b_t and c_t are in ext
      sorted_idx = sortu256_idx_h(a_t_u256)
      ecbn128_samples = np.concatenate((a_t_u256[sorted_idx],G1.as_uint256(G1)[:2]))
      #ecbn128_samples = np.concatenate((a_t_u256,G1.as_uint256(G1)[:2]))
      self.A,_ = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.A = ec_jac2aff_h(self.A.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.A = np.reshape(self.A,(-1,3,NWORDS_256BIT))[unsorted_idx]
      #self.A = np.reshape(self.A,(-1,3,NWORDS_256BIT))
      self.A = np.reshape(self.A,(-1,NWORDS_256BIT))

      sorted_idx = sortu256_idx_h(b_t_u256)
      ecbn128_samples[:-2] = b_t_u256[sorted_idx]
      self.B1,t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.B1 = ec_jac2aff_h(self.B1.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.B1 = np.reshape(self.B1,(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.B1=np.reshape(self.B1,(-1,NWORDS_256BIT))

      ec2bn128_samples = np.concatenate((b_t_u256[sorted_idx],G2.as_uint256(G2)[:4]))
      #ec2bn128_samples = np.concatenate((b_t_u256,G2.as_uint256(G2)[:4]))
      self.B2,t = ec_sc1mul_cuda(self.ec2bn128, ec2bn128_samples, ZField.get_field(), ec2=True)
      self.B2 = ec2_jac2aff_h(self.B2.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.B2 = np.reshape(self.B2,(-1,6,NWORDS_256BIT))[unsorted_idx]
      #self.B2 = np.reshape(self.B2,(-1,6,NWORDS_256BIT))
      self.B2 = np.reshape(self.B2,(-1,NWORDS_256BIT))
      #ECC.from_uint256(self.B2.reshape((-1,2,8))[0:3],reduced=True, in_ectype=2, out_ectype=2,ec2=True)[0].extend().as_list()

      ZField.set_field(MOD_FIELD)
      pidx = ZField.get_field()
      ps_u256 = GrothSetupComputePS_h(self.toxic['kalfa'].reduce().as_uint256(), self.toxic['kbeta'].reduce().as_uint256(),
                                      toxic_invDelta.reduce().as_uint256(),
                                      a_t_u256, b_t_u256, c_t_u256, self.nPublic, pidx )
      ZField.set_field(MOD_GROUP)
      sorted_idx = sortu256_idx_h(ps_u256)
      ecbn128_samples = np.concatenate((ps_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.C,t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.C = ec_jac2aff_h(self.C.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.C = np.reshape(self.C,(-1,3,NWORDS_256BIT))[unsorted_idx]
      #ECC.from_uint256(self.C[12],reduced=True, in_ectype=2, out_ectype=2)[0].extend().as_list()
      self.C=np.concatenate((np.zeros(((self.nPublic+1)*3,NWORDS_256BIT),dtype=np.uint32),np.reshape(self.C,(-1,NWORDS_256BIT))))


      maxH = self.domainSize+1;
      self.hExps = np.zeros((maxH,NWORDS_256BIT),dtype=np.uint32)

      ZField.set_field(MOD_FIELD)
      pidx = ZField.get_field()
      zod_u256 = montmult_h(toxic_invDelta.reduce().as_uint256(), z_t_u256, pidx)
      eT_u256 = GrothSetupComputeeT_h(self.toxic['t'].reduce().as_uint256(), zod_u256, maxH, pidx)

      ZField.set_field(MOD_GROUP)
      sorted_idx = sortu256_idx_h(eT_u256)
      ecbn128_samples = np.concatenate((eT_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.hExps,t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.hExps = ec_jac2aff_h(self.hExps.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.hExps = np.reshape(self.hExps,(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.hExps=np.reshape(self.hExps,(-1,NWORDS_256BIT))


      #TODO
      """
      for (let s=setup.vk_proof.nPublic+1; s<circuit.nVars; s++) {
        let ps =
            F.mul(
                invDelta,
                F.add(
                    F.add(
                        F.mul(v.a_t[s], setup.toxic.kbeta),
                        F.mul(v.b_t[s], setup.toxic.kalfa)),
                   v.c_t[s]));
       const C = G1.affine(G1.mulScalar(G1.g, ps));
       setup.vk_proof.C[s]=C;
       """

    def _calculateValuesAtT(self):
       # Required z_t, polsA/B/C are in Ext format
       # u is Mont
       z_t_u256, u = self._evalLagrangePoly(self.domainBits)

       pidx = ZField.get_field()

       a_t_u256 = mpoly_madd_h(self.polsA, u.reshape(-1), self.nVars, pidx)
       b_t_u256 = mpoly_madd_h(self.polsB, u.reshape(-1), self.nVars, pidx)
       c_t_u256 = mpoly_madd_h(self.polsC, u.reshape(-1), self.nVars, pidx)

       return a_t_u256, b_t_u256, c_t_u256, z_t_u256

    def _cirbin_to_u256(self):
        return readU256CircuitFile_h(self.in_circuit_f.encode("UTF-8"))

    def _ciru256_to_bin(self, ciru256_data):
        writeU256CircuitFile_h(ciru256_data, self.out_circuit_f.encode("UTF-8"))

    def _vars_to_toxicdict(self):
      toxic_dict = self.toxic
      return toxic_dic

    def _vars_to_pkdict(self):
        pk_dict={}
        pk_dict['protocol'] = "groth"
        pk_dict['field_p'] = str(ZFieldElExt.from_uint256(self.field_p).as_long())
        pk_dict['group_p'] = str(ZFieldElExt.from_uint256(self.group_p).as_long())
        if self.out_k_binformat == FMT_EXT:
           pk_dict['binFormat'] = "normal"
           b_reduce = False
        else:
           pk_dict['binFormat'] = "montgomery"
           b_reduce=True

        pk_dict['Rbitlen'] = int(self.Rbitlen)

        if self.out_k_ecformat == EC_T_AFFINE:
           pk_dict['ecFormat'] = "affine"
        elif self.out_k_ecformat == EC_T_JACOBIAN: 
           pk_dict['ecFormat'] = "jacobian"
        else :
           pk_dict['ecFormat'] = "projective"

           
        pk_dict['nVars'] = int(self.nVars)
        pk_dict['nPublic'] = int(self.nPublic)
        pk_dict['domainBits'] = int(self.domainBits)
        pk_dict['domainSize'] = int(self.domainSize)

        ZField.set_field(MOD_FIELD)
        if self.out_k_binformat == FMT_EXT:
          spoly = mpoly_to_sparseu256_h(self.polsA)
          pk_dict['polsA'] = [{k : str(ZFieldElRedc(BigInt.from_uint256(p[k])).extend().as_long()) for  k in p.keys()} for p in spoly]
          spoly = mpoly_to_sparseu256_h(self.polsB)
          pk_dict['polsB'] = [{k : str(ZFieldElRedc(BigInt.from_uint256(p[k])).extend().as_long()) for  k in p.keys()} for p in spoly]
          spoly = mpoly_to_sparseu256_h(self.polsC)
          pk_dict['polsC'] = [{k : str(ZFieldElRedc(BigInt.from_uint256(p[k])).extend().as_long()) for  k in p.keys()} for p in spoly]
        else:
          spoly = mpoly_to_sparseu256_h(self.polsA)
          pk_dict['polsA'] = [{k : str(BigInt.from_uint256(p[k]).as_long()) for  k in p.keys()} for p in spoly]
          spoly = mpoly_to_sparseu256_h(self.polsB)
          pk_dict['polsB'] = [{k : str(BigInt.from_uint256(p[k]).as_long()) for  k in p.keys()} for p in spoly]
          spoly = mpoly_to_sparseu256_h(self.polsC)
          pk_dict['polsC'] = [{k : str(BigInt.from_uint256(p[k]).as_long()) for  k in p.keys()} for p in spoly]


        ZField.set_field(MOD_GROUP)
        P = ECC.from_uint256(self.A, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)
        if not b_reduce:
           pk_dict['A'] = [x.extend().as_str() for x in P]
        else:
           pk_dict['A'] = [x.as_str() for x in P]

        P = ECC.from_uint256(self.B1, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)
        if not b_reduce:
          pk_dict['B1'] = [x.extend().as_str() for x in P]
        else:
          pk_dict['B1'] = [x.as_str() for x in P]

        P = ECC.from_uint256(self.B2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True, ec2=True)
        if not b_reduce:
          pk_dict['B2'] = [x.extend().as_str() for x in P]
        else:
          pk_dict['B2'] = [x.as_str() for x in P]

        P = ECC.from_uint256(self.C, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)
        if not b_reduce:
          pk_dict['C'] = [x.extend().as_str() for x in P]
        else:
          pk_dict['C'] = [x.as_str() for x in P]

        if not b_reduce:
          pk_dict['vk_alfa_1'] = ECC.from_uint256(self.vk_alfa_1, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)[0].extend().as_str()
          pk_dict['vk_beta_1'] = ECC.from_uint256(self.vk_beta_1, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)[0].extend().as_str()
          pk_dict['vk_delta_1'] = ECC.from_uint256(self.vk_delta_1, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)[0].extend().as_str()
          pk_dict['vk_beta_2'] = ECC.from_uint256(self.vk_beta_2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True, ec2=True)[0].extend().as_str()
          pk_dict['vk_delta_2'] = ECC.from_uint256(self.vk_delta_2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True, ec2=True)[0].extend().as_str()

        else:
          pk_dict['vk_alfa_1'] = ECC.from_uint256(self.vk_alfa_1, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)[0].as_str()
          pk_dict['vk_beta_1'] = ECC.from_uint256(self.vk_beta_1, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)[0].as_str()
          pk_dict['vk_delta_1'] = ECC.from_uint256(self.vk_delta_1, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)[0].as_str()
          pk_dict['vk_beta_2'] = ECC.from_uint256(self.vk_beta_2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True, ec2=True)[0].as_str()
          pk_dict['vk_delta_2'] = ECC.from_uint256(self.vk_delta_2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True, ec2=True)[0].as_str()

        P = ECC.from_uint256(self.hExps, in_ectype=EC_T_AFFINE, out_ectype=self.out_k_ecformat, reduced=True)
        if not b_reduce:
          pk_dict['hExps'] = [x.extend().as_str() for x in P]
        else:
          pk_dict['hExps'] = [x.as_str() for x in P]
       
        return pk_dict
 
           
    def _vars_to_pkbin(self):
        pk_bin = np.concatenate(
                   self.protocol,
                   self.Rbitlen,
                   self.field_p,
                   self.group_p,
                   self.out_k_binformat,
                   self.out_k_ecformat)

        if self.out_k_binformat == FMT_MONT:
           self.vk_alfa_1 = from_montgomeryN_h(self.vk_alfa_1, MOD_GROUP)
           self.vk_beta_1 = from_montgomeryN_h(self.vk_beta_1, MOD_GROUP)
           self.vk_delta_1 = from_montgomeryN_h(self.vk_delta_1, MOD_GROUP)
           self.vk_beta_2 = from_montgomeryN_h(self.vk_beta_2, MOD_GROUP)
           self.vk_delta_2 = from_montgomeryN_h(self.vk_delta_2, MOD_GROUP)
           self.A = from_montgomeryN_h(self.A, MOD_GROUP)
           self.B1 = from_montgomeryN_h(self.B1, MOD_GROUP)
           self.B2 = from_montgomeryN_h(self.B2, MOD_GROUP)
           self.C = from_montgomeryN_h(self.C, MOD_GROUP)
           self.hExps = from_montgomeryN_h(self.hExps, MOD_GROUP)
           self.polsA = mpoly_from_montgomery_h(self.polsA, MOD_FIELD)
           self.polsB = mpoly_from_montgomery_h(self.polsB, MOD_FIELD
           self.polsC = mpoly_from_montgomery_h(self.polsC, MOD_FIELD)
        
        pk_bin = np.concatenate(
                  pk_bin,
                  self.nVars,
                  self,nPublic,
                  self.domainSize,
                  self.vk_alfa_1,    
                  self.vk_beta_1,    
                  self.vk_delta_1,   
                  self.vk_beta_2,    
                  self.vk_delta_2,   
                  np.asarray(self.polsA.shape[0],dtype=np.uint32),
                  self.polsA,
                  np.asarray(self.polsB.shape[0],dtype=np.uint32),
                  self.polsB,
                  np.asarray(self.polsC.shape[0],dtype=np.uint32),
                  self.polsC,
                  np.asarray(self.A.shape[0],dtype=np.uint32),
                  self.A,
                  np.asarray(self.B1.shape[0],dtype=np.uint32),
                  self.B1,
                  np.asarray(self.B2.shape[0],dtype=np.uint32),
                  self.B2,
                  np.asarray(self.C.shape[0],dtype=np.uint32),
                  self.C,
                  np.asarray(self.hExps.shape[0],dtype=np.uint32),
                  self.hExps)
       
        return pk_bin


    def _ciru256_to_vars(self, ciru256_data):
        R1CSA_offset = CIRBIN_H_N_OFFSET
        R1CSB_offset = CIRBIN_H_N_OFFSET +  \
                       np.uint32(ciru256_data[CIRBIN_H_CONSTA_NWORDS_OFFSET])
        R1CSC_offset = CIRBIN_H_N_OFFSET + \
                       np.uint32(ciru256_data[CIRBIN_H_CONSTA_NWORDS_OFFSET]) + \
                       np.uint32(ciru256_data[CIRBIN_H_CONSTB_NWORDS_OFFSET])

        self.nWords        =  np.uint32(ciru256_data[CIRBIN_H_NWORDS_OFFSET])
        self.nPubInputs    =  np.uint32(ciru256_data[CIRBIN_H_NPUBINPUTS_OFFSET])
        self.nOutputs      =  np.uint32(ciru256_data[CIRBIN_H_NOUTPUTS_OFFSET])
        self.nVars         =  np.uint32(ciru256_data[CIRBIN_H_NVARS_OFFSET])
        self.nConstraints  =  np.uint32(ciru256_data[CIRBIN_H_NCONSTRAINTS_OFFSET])
        self.cirformat       =  np.uint32(ciru256_data[CIRBIN_H_FORMAT_OFFSET])
        self.R1CSA_nWords =  np.uint32(ciru256_data[CIRBIN_H_CONSTA_NWORDS_OFFSET])
        self.R1CSB_nWords =  np.uint32(ciru256_data[CIRBIN_H_CONSTB_NWORDS_OFFSET])
        self.R1CSC_nWords =  np.uint32(ciru256_data[CIRBIN_H_CONSTC_NWORDS_OFFSET])
        self.R1CSA        =  ciru256_data[R1CSA_offset:R1CSB_offset] 
        self.R1CSB        =  ciru256_data[R1CSB_offset:R1CSC_offset]
        self.R1CSC        =  ciru256_data[R1CSC_offset:] 

    def _cirvarsPack(self):
        return  np.concatenate((
                       [self.nWords,
                        self.nPubInputs,
                        self.nOutputs,
                        self.nVars,
                        self.nConstraints,
                        self.cirformat,
                        self.R1CSA_nWords,
                        self.R1CSB_nWords,
                        self.R1CSC_nWords],
                        self.R1CSA,
                        self.R1CSB,
                        self.R1CSC))

    
    def _computeHeader(self):

        self.header = {'nWords' : self.nWords,
                  'nPubInputs' : self.nPubInputs,
                  'nOutputs' : self.nOutputs,
                  'nVars' : self.nVars,
                  'nConstraints' : self.nConstraints,
                  'cirformat' : self.cirformat,
                  'R1CSA_nWords' : self.R1CSA_nWords,
                  'R1CSB_nWords' : self.R1CSB_nWords,
                  'R1CSC_nWords' : self.R1CSC_nWords}

    def _cirjson_to_u256(self,circuit_f):
        """
          Converts from circom .json output file to binary format required to 
            calculate snarks setup. Only the following entries are used:
             - constraints -> R1CS a,b,c
             - nPubInputs  -> k
             - nVars       -> N
             - nOutputs    ->
             - cirformat      -> EXT[0]/MONT[1]

          R1CS binary format:
            N constraints  -------------------------------- 32 bits  
            cumsum(  -> cumulative
              N coeff constraints[0] ---------------------- 32 bits
              N coeff constraints[1] ---------------------- 32 bits : N constraints[0] + N constraints[1]
              ----
              N coeff constraints[N-1] -------------------- 32 bits : N contraints[0] + N constraints[1] +
                                                                      N constraints[2] +...+ Nconstraints[N-1]
            )
            Coeff[0,0] constraint 0, coeff 0 -------------- 32 bits
            Coeff[0,1] constraint 0, coeff 1 -------------- 32 bits
            ----
            Coeff[0,C0-1] constraint 0, coeff C0-1 -------- 32 bits
            Val[0,0] constraint 0, value 0 ---------------- 256 bits (8 words) : word 0 is LSW
            Val[0,1] constraint 0, value 1 ---------------- 256 bits 
            ----
            Val[0,C0-1] constraint 0, value C0-1 - -------- 256 bits 
            Coeff[1,0] constraint 1, coeff 0 -------------- 32 bits
            Coeff[1,1] constraint 1, coeff 1 -------------- 32 bits
            ----
            Coeff[1,C1-1] constraint 1, coeff C1-1 -------- 32 bits
            Val[1,0] constraint 1, value 0 ---------------- 256 bits 
            Val[1,1] constraint 1, value 1 ---------------- 256 bits 
            ----
            Val[1,C1-1] constraint 1, value C1-1 -- ------- 256 bits 
            ----
            ----
            Coeff[N-1,0] constraint N-1, coeff 0 ---------- 32 bits
            Coeff[N-1,1] constraint N-1, coeff 1 ---------- 32 bits
            ----
            Coeff[N-1,CN_1-1] constraint N-1, coeff CN_1-1  32 bits
            Val[N-1,0] constraint 1, value 0 -------------- 256 bits 
            Val[N-1,1] constraint 1, value 1 -------------- 256 bits 
            ----
            Val[N-1,CN_1-1] constraint 1, value CN_1-1 ---- 256 bits 

          Binary file format
            nWords : File size in 32 bit workds --------------- 32 bits
            nPubInputs : -------------------------------------- 32 bits
            nOutputs   : -------------------------------------- 32 bits
            nVars      : -------------------------------------- 32 bits
            nConstraints : Number of constraints--------------- 32 bits
            cirformat : Extended[0]/Montgomery[1]----------------- 32 bits
            R1CSA_nWords : R1CSA size in 32 bit words --------- 32 bits
            R1CSB_nWords : R1CSB size in 32 bit words --------- 32 bits
            R1CSC_nWords : R1CSC size in 32 bit words --------- 32 bits
            R1CSA        :  R1CS  format 
            R1CSB        :  R1CS format
            R1CSC        : R1Cs format
 
            
        """
        labels = ['constraints', 'nPubInputs','nOutputs','nVars']
        f = open(circuit_f,'r')
        cir_json_data = json.load(f)
        cir_data = json_to_dict(cir_json_data, labels)
        f.close()

        if 'cirformat' in cir_data:
            self.in_circuit_format = cir_data['cirformat']

        if self.in_circuit_format == self.out_circuit_format:
          R1CSA_u256 = [ZPolySparse(coeff[0]).as_uint256() for coeff in cir_data['constraints']]
        elif self.in_circuit_format == ZUtils.FEXT:
          R1CSA_u256 = [ZPolySparse(coeff[0]).reduce().as_uint256() for coeff in cir_data['constraints']]
        else :
          R1CSA_u256 = [ZPolySparse(coeff[0]).extend().as_uint256() for coeff in cir_data['constraints']]

        R1CSA_l = []
        R1CSA_p = []
        for l,p in R1CSA_u256:
            R1CSA_l.append(l)
            R1CSA_p.append(p)
        R1CSA_u256 = np.asarray(np.concatenate((np.asarray([len(R1CSA_l)]),
                                              np.concatenate(
                                                 (np.cumsum(R1CSA_l), 
                                                  np.concatenate(R1CSA_p))))),
                                                  dtype=np.uint32)
        R1CSA_len = R1CSA_u256.shape[0]
                

        if self.in_circuit_format == self.out_circuit_format:
          R1CSB_u256 = [ZPolySparse(coeff[1]).as_uint256() for coeff in cir_data['constraints']]
        elif self.in_circuit_format == ZUtils.FEXT:
          R1CSB_u256 = [ZPolySparse(coeff[1]).reduce().as_uint256() for coeff in cir_data['constraints']]
        else :
          R1CSB_u256 = [ZPolySparse(coeff[1]).extend().as_uint256() for coeff in cir_data['constraints']]

        R1CSB_l = []
        R1CSB_p = []
        for l,p in R1CSB_u256:
            R1CSB_l.append(l)
            R1CSB_p.append(p)
        R1CSB_u256 = np.asarray(np.concatenate((np.asarray([len(R1CSB_l)]),
                                              np.concatenate(
                                                 (np.cumsum(R1CSB_l), 
                                                  np.concatenate(R1CSB_p))))),
                                                  dtype=np.uint32)
        R1CSB_len = R1CSB_u256.shape[0]

        if self.in_circuit_format == self.out_circuit_format:
          R1CSC_u256 = [ZPolySparse(coeff[2]).as_uint256() for coeff in cir_data['constraints']]
        elif self.in_circuit_format == ZUtils.FEXT:
          R1CSC_u256 = [ZPolySparse(coeff[2]).reduce().as_uint256() for coeff in cir_data['constraints']]
        else :
          R1CSC_u256 = [ZPolySparse(coeff[2]).extend().as_uint256() for coeff in cir_data['constraints']]
    

        R1CSC_l = []
        R1CSC_p = []
        for l,p in R1CSC_u256:
            R1CSC_l.append(l)
            R1CSC_p.append(p)
        R1CSC_u256 = np.asarray(np.concatenate((np.asarray([len(R1CSC_l)]),
                                              np.concatenate(
                                                 (np.cumsum(R1CSC_l), 
                                                  np.concatenate(R1CSC_p))))),
                                                  dtype=np.uint32)
        R1CSC_len = R1CSC_u256.shape[0]

        fsize = CIRBIN_H_N_OFFSET + R1CSA_len + R1CSB_len + R1CSC_len

        self.nWords       =  np.uint32(fsize)
        self.nPubInputs   =  np.uint32(cir_data['nPubInputs'])
        self.nOutputs     =  np.uint32(cir_data['nOutputs'])
        self.nVars        =  np.uint32(cir_data['nVars'])
        self.nConstraints =  np.uint32(len(cir_data['constraints']))
        self.cirformat       =  np.uint32(self.out_circuit_format)
        self.R1CSA_nWords =  np.uint32(R1CSA_len)
        self.R1CSB_nWords =  np.uint32(R1CSB_len)
        self.R1CSC_nWords =  np.uint32(R1CSC_len)
        self.R1CSA        =  R1CSA_u256
        self.R1CSB        =  R1CSB_u256
        self.R1CSC        =  R1CSC_u256 

        return  self._cirvarsPack()

    def write_pk(self):
       out_pk_f = self.out_pk_f
       if self.test_f is not None:
         out_pk_f = './data/xxx.json'

       if out_pk_f.endswith('.json') :
         pk_dict = self._vars_to_pkdict()
         pk_json = json.dumps(pk_dict, indent=4, sort_keys=True)
         f = open(self.out_pk_f, 'w')
         print(pk_json, file=f)
         f.close()

       if self.out_pk_f.endswith('bin') :
         pk_bin = self._vars_to_pkbin()
         writeU256CircuitFile_h(pk_bin, self.out_pk_f.encode("UTF-8"))


    def write_toxic(self)
       toxic_dict = self._vars_to_toxicdict()
       toxic_json = json.dumps(toxic_dic, indent=4, sort_keys=True)
       f = open(self.test_f, 'w')
       print(toxic_json, file=f)
       f.close()

    def write_vk(self):
       self._gen_vk_alfabeta_12()
       if self.out_vk_f.endswith('.json') :
         vk_dict = self._vars_to_vkdict()
         pk_json = json.dumps(vk_dict, indent=4, sort_keys=True)
         f = open(self.out_vk_f, 'w')
         print(pk_json, file=f)
         f.close()

       elif self.out_vk_f.endswith('bin') :
         vk_bin = self._vars_to_vkbin()

         if vk_bin is not None:
            writeU256CircuitFile_h(vk_bin, self.out_vk_f.encode("UTF-8"))

    def _vars_to_vkdict(self):
      # TODO : only suported formats for vk are .json, affine and extended 
      vk_dict = {}
      vk_dict['protocol'] = "groth"
      vk_dict['field_p'] = str(ZFieldElExt.from_uint256(self.field_p).as_long())
      vk_dict['group_p'] = str(ZFieldElExt.from_uint256(self.group_p).as_long())
      vk_dict['binFormat'] = "normal"

      vk_dict['Rbitlen'] = int(self.Rbitlen)
      vk_dict['ecFormat'] = "affine"

      vk_dict['nVars'] = int(self.nVars)
      vk_dict['nPublic'] = int(self.nPublic)
      vk_dict['domainBits'] = int(self.domainBits)
      vk_dict['domainSize'] = int(self.domainSize)

      ZField.set_field(MOD_FIELD)
      vk_dict['vk_alfa_1'] = ECC.from_uint256(self.vk_alfa_1, in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)[0].extend().as_str()
      vk_dict['vk_beta_1'] = ECC.from_uint256(self.vk_beta_1, in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)[0].extend().as_str()
      vk_dict['vk_delta_1'] = ECC.from_uint256(self.vk_delta_1, in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)[0].extend().as_str()
      vk_dict['vk_beta_2'] = ECC.from_uint256(self.vk_beta_2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True, ec2=True)[0].extend().as_str()
      vk_dict['vk_delta_2'] = ECC.from_uint256(self.vk_delta_2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True, ec2=True)[0].extend().as_str()

      ZField.set_field(MOD_GROUP)
      P = ECC.from_uint256(self.IC, in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)
      vk_dict['IC'] = [x.extend().as_str() for x in P]

      return vk_dict

    def _vars_to_vkbin(self):
      print("Verifying Key can only be saved as .json\n");
      vk_bin=None
      return vk_bin


if __name__ == "__main__":
    in_circuit_f = '../../data/prove-kyc.json'
    out_circuit_f = '../../data/prove-kyc.bin'
    #in_circuit_f = '../../data/circuit.json'
    #out_circuit_f = '../../data/circuit.bin'
    out_pk_f = '../../data/proving_key_prove-kyc.json'

    #GS = GrothSetup(in_circuit_f = in_circuit_f, out_circuit_f = out_circuit_f,
    GS = GrothSetup(in_circuit_f = out_circuit_f,
                    out_circuit_format=FMT_EXT, out_pk_f=out_pk_f, 
                    out_k_binformat=FMT_EXT, out_k_ecformat=EC_T_AFFINE, test_f=None)
    """
    if os.path.isfile(out_circuit_f):
       GS = GrothSetup(in_circuit_f=out_circuit_f, 
                       out_pk_f=out_pk_f, out_k_binformat=FMT_MONT,
                       out_k_ecformat=EC_T_AFFINE, test_f=toxic_vals)
    else:
       GS = GrothSetup(in_circuit_f=in_circuit_f, out_circuit_f=out_circuit_f, 
                       out_circuit_format=FMT_MONT, out_pk_f=out_pk_f, out_k_binformat=FMT_MONT,
                       out_k_ecformat=EC_T_AFFINE, test_f=toxic_vals)
    """

    GS.setup()

   

