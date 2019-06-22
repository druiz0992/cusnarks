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


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

ROOTS_1M_filename_npz = '../../data/zpoly_data_1M.npz'
ROOTS_1M_filename_bin = '../../data/zpoly_roots_1M.bin'

class GrothSetup(object):
    
    GroupIDX = 0
    FieldIDX = 1

    def __init__(self, curve='BN128', in_circuit_f=None, out_circuit_f=None, in_circuit_format=FMT_EXT, out_circuit_format=FMT_MONT, out_pk_f=None, out_pk_binformat=FMT_MONT, out_pk_ecformat=EC_T_AFFINE, toxic_k=None):
  
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data)
        ZPoly.init(GrothSetup.GroupIDX)
        self.group_p        = ZField.get_extended_p().as_uint256()

        ZPoly.init(GrothSetup.FieldIDX)

        self.field_p        = ZField.get_extended_p().as_uint256()
   
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

        self.ecbn128    = None 
        self.ec2bn128   = None 

        self.out_pk_f   = out_pk_f
        self.out_pk_binformat = out_pk_binformat
        self.out_pk_ecformat = out_pk_ecformat

        self.toxic_k = toxic_k

        if in_circuit_f is not None:
           self.circuitRead(in_circuit_f, in_circuit_format, out_circuit_format, out_circuit_f)

    def circuitRead(self,in_circuit_f, in_circuit_format, out_circuit_format, out_circuit_f=None):
        # cir Json to u256
        if in_circuit_f.endswith('.json'):
           cir_u256 = self._cirjson_to_u256(in_circuit_f, in_circuit_format=in_circuit_format, out_circuit_format=out_circuit_format)

           #u256 to bin
           if out_circuit_f is not None:
              self._ciru256_to_bin(cir_u256, out_circuit_f)

        elif in_circuit_f.endswith('.bin'):
             cir_u256 = self._cirbin_to_u256(in_circuit_f)

        self._ciru256_to_vars(cir_u256)

    def setup(self):
        ZPoly.init(GrothSetup.FieldIDX)
        self.domainBits =  np.uint32(math.ceil(math.log(self.nConstraints+ 
                                           self.nPubInputs + 
                                           self.nOutputs,2)))


        self.nPublic    = self.nPubInputs + self.nOutputs
        self.domainSize = 1 << self.domainBits

        prime = ZField.get_extended_p()
        if toxic_k is None:
           toxic_trdc = ZFieldElRedc(randint(1,prime.as_long()-1))
        else:
           toxic_trdc = ZFieldElExt(toxic_k['t']).reduce()

        self._calculatePoly()
        self._calculateEncryptedValuesAtT(toxic_trdc)

        return 


    def _calculatePoly(self):
        self._computeHeader()

        polyA_len = r1cs_to_mpoly_len_h(self.R1CSA, self.header, 1)
        self.polsA = r1cs_to_mpoly_h(polyA_len,self.R1CSA, self.header, 1)
        self.R1CSA = None

        polyB_len = r1cs_to_mpoly_len_h(self.R1CSB, self.header, 0)
        self.polsB = r1cs_to_mpoly_h(polyB_len,self.R1CSB, self.header, 0)
        self.R1CSB = None

        polyC_len = r1cs_to_mpoly_len_h(self.R1CSC, self.header, 0)
        self.polsC = r1cs_to_mpoly_h(polyC_len,self.R1CSC, self.header, 0)
        self.R1CSC = None

    def _evalLagrangePoly(self, bits, t_rdc):
       """
        m : int
        t : ZFieldElRedc
       """
       m = 1 << bits
       tm = (t_rdc ** int(m))
       u_u256 = np.zeros((m,NWORDS_256BIT),dtype=np.uint32)
       trdc_u256 = t_rdc.as_uint256()
      
       #load roots
       if os.path.exists(ROOTS_1M_filename_npz):
           npzfile = np.load(ROOTS_1M_filename_npz)
           roots_rdc_u256 = npzfile['roots_rdc_u256'][::1<<(20-bits)]
       else :
           roots_rdc_u256, _ = ZField.find_roots(m, find_inv_roots = False, rformat_ext=False)
           roots_rdc_u256 = roots_rdc_u256[::1<<(20-bits)]
           roots_rdc_u256 = np.asarray([r.as_uint256() for r in roots_rdc_u256],dtype=np.uint32)

       omega = ZFieldElRedc(BigInt.from_uint256(roots_rdc_u256[1]))

       z = tm - 1
       z_rdc = z.reduce()
       if tm == ZFieldElExt(1).reduce():
         for i in xrange(m): 
           #TODO : roots[0] is always 1. Does this make any sense? check javascript version
           if roots_rdc_u256[0] == trdc_u256:
             u_u256[i] = ZFieldElExt(1).reduce().as_uint256()
             return z_rdc, u_u256()

       l_rdc = z_rdc * ZFieldElExt(int(m)).inv().reduce()
       lrdc_u256 = l_rdc.as_uint256()

       pidx = ZField.get_field()
       u_u256 = evalLagrangePoly_h(trdc_u256,lrdc_u256, roots_rdc_u256, pidx)

       return z_rdc, u_u256
   
    def _calculateEncryptedValuesAtT(self, toxic_trdc):
      a_t_u256, b_t_u256, c_t_u256, z_t = self._calculateValuesAtT(toxic_trdc)

      prime = ZField.get_extended_p()
      curve_params = self.curve_data['curve_params']
      curve_params_g2 = self.curve_data['curve_params_g2']

      # Tocix k extended
      if toxic_k is None:
        toxic_kalfa = ZFieldElExt(randint(1,prime.as_long()-1))
        toxic_kbeta = ZFieldElExt(randint(1,prime.as_long()-1))
        toxic_kgamma = ZFieldElExt(randint(1,prime.as_long()-1))
        toxic_kdelta = ZFieldElExt(randint(1,prime.as_long()-1))

      else:
        toxic_kalfa = ZFieldElExt(toxic_k['alfa'])
        toxic_kbeta = ZFieldElExt(toxic_k['beta'])
        toxic_kgamma = ZFieldElExt(toxic_k['gamma'])
        toxic_kdelta = ZFieldElExt(toxic_k['delta'])

      toxic_invDelta = toxic_kdelta.inv()


      ZPoly.init(GrothSetup.GroupIDX)
      Gx = ZFieldElExt(curve_params['Gx'])
      Gy = ZFieldElExt(curve_params['Gy'])
      G2x = Z2FieldEl([curve_params_g2['Gx1'], curve_params_g2['Gx2']])
      G2y = Z2FieldEl([curve_params_g2['Gy1'], curve_params_g2['Gy2']])

      G1 = ECCAffine([Gx,Gy]).reduce()
      G2 = ECCAffine([G2x, G2y]).reduce()

      # vk coeff MONT
      self.vk_alfa_1 = G1 * toxic_kalfa
      self.vk_beta_1 = G1 * toxic_kbeta
      self.vk_delta_1 = G1 * toxic_kdelta

      self.vk_alfa_1 = ec_jac2aff_h(G1.as_uint256(self.vk_alfa_1).reshape(-1),ZField.get_field())
      self.vk_beta_1 = ec_jac2aff_h(G1.as_uint256(self.vk_beta_1).reshape(-1),ZField.get_field())
      self.vk_delta_1 = ec_jac2aff_h(G1.as_uint256(self.vk_delta_1).reshape(-1),ZField.get_field())
      
      self.vk_beta_2 = G2 * toxic_kbeta
      self.vk_delta_2 = G2 * toxic_kdelta
    
      self.vk_beta_2 = ec2_jac2aff_h(G2.as_uint256(self.vk_beta_2).reshape(-1),ZField.get_field())
      self.vk_delta_2 = ec2_jac2aff_h(G2.as_uint256(self.vk_delta_2).reshape(-1),ZField.get_field())

      self.ecbn128    =  ECBN128(self.domainSize+3,seed=1)
      self.ec2bn128    = EC2BN128(self.nVars+1,seed=1)
   
      # a_t, b_t and c_t are in montgomery. I converted them to extended in Cuda to do multiplication.
      # TODO : check that sort function maintains order
      sorted_idx = sortu256_idx_h(a_t_u256)
      ecbn128_samples = np.concatenate((a_t_u256[sorted_idx],G1.as_uint256(G1)[:2]))
      self.A,_ = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.A = ec_jac2aff_h(self.A.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.A = np.reshape(self.A,(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.A = np.reshape(self.A,(-1,NWORDS_256BIT))

      sorted_idx = sortu256_idx_h(b_t_u256)
      ecbn128_samples[:-2] = b_t_u256[sorted_idx]
      self.B1,t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.B1 = ec_jac2aff_h(self.B1.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.B1 = np.reshape(self.B1,(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.B1=np.reshape(self.B1,(-1,NWORDS_256BIT))

      ec2bn128_samples = np.concatenate((b_t_u256[sorted_idx],G2.as_uint256(G2)[:4]))
      self.B2,t = ec2_sc1mul_cuda(self.ec2bn128, ec2bn128_samples, ZField.get_field())
      self.B2 = ec2_jac2aff_h(self.B2.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.B2 = np.reshape(self.B2,(-1,6,NWORDS_256BIT))[unsorted_idx]
      self.B2 = np.reshape(self.B2,(-1,NWORDS_256BIT))

      ZPoly.init(GrothSetup.FieldIDX)
      pidx = ZField.get_field()
      ps_u256 = GrothSetupComputePS_h(toxic_kalfa.reducce().as_uint256(), toxic_kbeta.reduce().as_uint256(),
                                      toxic_invDelta.reduce().as_uint256(),
                               a_t_u256, b_t_u256, c_t_u256, self.nPublic, pidx )
      ZPoly.init(GrothSetup.GroupIDX)
      sorted_idx = sortu256_idx_h(ps_u256)
      ecbn128_samples = np.concatenate((ps_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.C,t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.C = ec_jac2aff_h(self.C.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.C = np.reshape(self.C,(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.C=np.reshape(self.C,(-1,NWORDS_256BIT))


      maxH = self.domainSize+1;
      self.hExps = np.zeros((maxH,NWORDS_256BIT),dtype=np.uint32)

      ZPoly.init(GrothSetup.FieldIDX)
      pidx = ZField.get_field()
      zod_u256 = montmult_h(toxic_invDelta.reduce().as_uint256(), z_t.as_uint256(), pidx)
      eT_u256 = GrothSetupComputeeT_h(toxic_trdc.as_uint256(), zod_u256, maxH, pidx)

      ZPoly.init(GrothSetup.GroupIDX)
      sorted_idx = sortu256_idx_h(eT_u256)
      ecbn128_samples = np.concatenate((eT_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.hExps,t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.hExps = ec_jac2aff_h(self.hExps.reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.hExps = np.reshape(self.hExps,(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.hExps=np.reshape(self.hExps,(-1,NWORDS_256BIT))


    def _calculateValuesAtT(self, toxic_trdc):
       # Required z_t, u, polsA/B/C are in montgomery format
       z_t, u = self._evalLagrangePoly(self.domainBits, toxic_trdc)

       pidx = ZField.get_field()

       a_t_u256 = mpoly_madd_h(self.polsA, u, self.nVars, pidx)
       b_t_u256 = mpoly_madd_h(self.polsB, u, self.nVars, pidx)
       c_t_u256 = mpoly_madd_h(self.polsC, u, self.nVars, pidx)

       return a_t_u256, b_t_u256, c_t_u256, z_t

    def _cirbin_to_u256(self, circuit_f):
        return readU256CircuitFile_h(circuit_f.encode("UTF-8"))

    def _ciru256_to_bin(self, ciru256_data, circuit_f):
        writeU256CircuitFile_h(ciru256_data, circuit_f.encode("UTF-8"))

    def _vars_to_pkdict(self):
        pk_dict={}
                   self.ec_format)
        pk_dict['protocol'] = "groth"
        pk_dict['field_p'] = str(ZFieldElExt.from_uint256(self.field_p))
        pk_dict['group_p'] = str(ZFieldElExt.from_uint256(self.group_p))
        if self.out_pk_binformat == FMT_EXT:
           pk_dict['binFormat'] = "normal"
        else:
           pk_dict['binFormat'] = "montgomery"
        pk_dict['Rbitlen'] = str(Zfield.get_reduction_data()['Rbitlen'])

        if self.out_pk_ecformat == EC_T_AFFINE:
           pk_dict['ecFormat'] = "affine"
        elif self.out_pk_ecformat == EC_T_JACOBIAN: 
           pk_dict['ecFormat'] = "jacobian"
        else :
           pk_dict['ecFormat'] = "projective"

           
        pk_dict['nVars'] = str(self.nVars)
        pk_dict['nPublic'] = str(self.nPublic)
        pk_dict['domainBits'] = str(self.domainBits)
        pk_dict['domainSize'] = str(self.domainSize)

        if self.out_pk_binformat == FMT_MONT:
           b_reduce=True
        else:
           b_reduce = False
  
        if self.out_pk_binformat == FMT_EXT:
          spoly = mpoly_to_sparseu256_h(self.polsA)
          pk_dict['polsA'] = [{k : ZFieldElRedc.from_uint256(p[k]).extend().as_long() for  k in p.keys()} for p in spoly]
          spoly = mpoly_to_sparseu256_h(self.polsB)
          pk_dict['polsB'] = [{k : ZFieldElRedc.from_uint256(p[k]).extend().as_long() for  k in p.keys()} for p in spoly]
          spoly = mpoly_to_sparseu256_h(self.polsC)
          pk_dict['polsC'] = [{k : ZFieldElRedc.from_uint256(p[k]).extend().as_long() for  k in p.keys()} for p in spoly]
        else:
          spoly = mpoly_to_sparseu256_h(self.polsA)
          pk_dict['polsA'] = [{k : BigInt.from_uint256(p[k]).as_long() for  k in p.keys()} for p in spoly]
          spoly = mpoly_to_sparseu256_h(self.polsB)
          pk_dict['polsB'] = [{k : BigInt.from_uint256(p[k]).as_long() for  k in p.keys()} for p in spoly]
          spoly = mpoly_to_sparseu256_h(self.polsC)
          pk_dict['polsC'] = [{k : BigInt.from_uint256(p[k]).as_long() for  k in p.keys()} for p in spoly]


        P = ECC.from_uint256(self.A, in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce)
        pk_dict['A'] = [x.as_list() for x in P]
        P = ECC.from_uint256(self.B1, in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce)
        pk_dict['B1'] = [x.as_list() for x in P]
        P = ECC.from_uint256(self.B2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce, ec2=True)
        pk_dict['B2'] = [x.as_list() for x in P]
        P = ECC.from_uint256(self.C, in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce)
        pk_dict['C'] = [x.as_list() for x in P]

        pk_dict['vk_alfa_1'] = ECC.from_uint256(self.vk_alfa_1, in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce)[0].as_list()
        pk_dict['vk_beta_1'] = ECC.from_uint256(self.vk_beta_1, in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce)[0].as_list()
        pk_dict['vk_delta_1'] = ECC.from_uint256(self.vk_delta_1, in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce)[0].as_list()
        pk_dict['vk_beta_2'] = ECC.from_uint256(self.vk_beta_2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce, ec2=True)[0].as_list()
        pk_dict['vk_delta_2'] = ECC.from_uint256(self.vk_delta_2.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce, ec2=True)[0].as_list()

        P = ECC.from_uint256(self.hExps, in_ectype=EC_T_AFFINE, out_ectype=self.out_pk_ecformat, reduced=b_reduce)
        pk_dict['hExps'] = [x.as_list() for x in P]
       
        return pk_dict
 

    def _vars_to_pkbin(self):
        pk_bin = np.concatenate(
                   self.protocol,
                   self.field_p,
                   self.group_p,
                   self.pk_format,
                   np.asarray(Zfield.get_reduction_data()['Rbitlen'],dtype=np.uint32),
                   self.ec_format)

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

    def _cirjson_to_u256(self,circuit_f, in_circuit_format=ZUtils.FEXT, out_circuit_format=ZUtils.FEXT):
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

        if in_circuit_format == out_circuit_format:
          R1CSA_u256 = [ZPolySparse(coeff[0]).as_uint256() for coeff in cir_data['constraints']]
        elif in_circuit_format == ZUtils.FEXT:
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
                

        if in_circuit_format == out_circuit_format:
          R1CSB_u256 = [ZPolySparse(coeff[1]).as_uint256() for coeff in cir_data['constraints']]
        elif in_circuit_format == ZUtils.FEXT:
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

        if in_circuit_format == out_circuit_format:
          R1CSC_u256 = [ZPolySparse(coeff[2]).as_uint256() for coeff in cir_data['constraints']]
        elif in_circuit_format == ZUtils.FEXT:
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
        self.cirformat       =  np.uint32(out_circuit_format)
        self.R1CSA_nWords =  np.uint32(R1CSA_len)
        self.R1CSB_nWords =  np.uint32(R1CSB_len)
        self.R1CSC_nWords =  np.uint32(R1CSC_len)
        self.R1CSA        =  R1CSA_u256
        self.R1CSB        =  R1CSB_u256
        self.R1CSC        =  R1CSC_u256 

        return  self._cirvarsPack()

    def write_pk(self):
       if self.out_pk_f.endswith('.json') :
         pk_dict = self._vars_to_pkdict()
         pk_json = json.dumps(pk_dict, indent=4).encode('utf8')
         f = open(self.out_pk_f, 'w')
         print(j, file=f)
         f.close()

       elif self.out_pk_f.endswith('bin') :
         pk_bin = self._vars_to_pkbin()
         writeU256CircuitFile_h(pk_bin, self.out_pk_f.encode("UTF-8"))


if __name__ == "__main__":
    in_circuit_f = '../../data/prove-kyc.json'
    out_circuit_f = '../../data/prove-kyc.bin'
    #in_circuit_f = '../../data/circuit.json'
    #out_circuit_f = '../../data/circuit.bin'
    out_pk_f = '../../data/proving_key_prove-kyc.json'
    if os.path.isfile(out_circuit_f):
       GS = GrothSetup(in_circuit_f=out_circuit_f)
    else:
       GS = GrothSetup(in_circuit_f=in_circuit_f, out_circuit_f=out_circuit_f, in_circuit_format=FMT_EXT, out_circuit_format=FMT_MONT, out_pk_f=out_pk_f, out_pk_binformat=FMT_MONT, out_pk_ecformat=EC_T_AFFINE, toxic_k=None)

    GS.setup()

    GS.write_pk()
