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
#from cuda_wrapper import *
from pysnarks_utils import *


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

ROOTS_1M_filename = '../../data/zpoly_data_1M.npz'

class GrothSetup(object):
    
    GroupIDX = 0
    FieldIDX = 1

    def __init__(self, curve='BN128', in_circuit_f=None, out_circuit_f=None, in_format=ZUtils.FEXT, out_format=ZUtils.FEXT):
  
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data['curve_params'])
        ZPoly.init(GrothSetup.FieldIDX)

        self.nWords       = None
        self.nPubInputs   = None
        self.nOutputs     = None
        self.nVars        = None
        self.nConstraints = None
        self.tformat       = None
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


        if in_circuit_f is not None:
           self.circuitRead(in_circuit_f, in_format, out_format, out_circuit_f)

    def circuitRead(self,in_circuit_f, in_format, out_format, out_circuit_f=None):
        # cir Json to u256
        if in_circuit_f.endswith('.json'):
           cir_u256 = self._cirjson_to_u256(in_circuit_f, in_format=in_format, out_format=out_format)

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


        self.nPublic    = self.nPubInputs + self.nOutputs,
        self.domainSize = 1 << self.domainBits

        prime = ZField.get_extended_p()
        toxic_t = ZFieldElRedc(randint(1,prime.as_long()-1))

        self._calculatePoly()
        a_t, b_t, c_t, z_t = self._calculateEncryptedValuesAtT(toxic_t)

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

    def _evalLagrangePoly(self, bits, t):
       """
        m : int
        t : ZFieldElRedc
       """
       m = 1 << bits
       tm = (t ** int(m))
       u_u256 = np.zeros((m,NWORDS_256BIT),dtype=np.uint32)
       t_u256 = t.as_uint256()
      
       #TODO : slice to get only m roots
       if os.path.exists(ROOTS_1M_filename):
           npzfile = np.load(ROOTS_1M_filename)
           roots_rdc_u256 = npzfile['roots_rdc_u256'][::1<<(20-bits)]
       else :
           roots_rdc_u256, _ = ZField.find_roots(m, find_inv_roots = False, rformat_ext=False)
           roots_rdc_u256 = roots_rdc_u256[::1<<(20-bits)]
           roots_rdc_u256 = np.asarray([r.as_uint256() for r in roots_rdc_u256],dtype=np.uint32)

       omega = ZFieldElRedc(BigInt.from_uint256(roots_rdc_u256[1]))

       z = tm - 1
       if tm == ZFieldElExt(1).reduce():
         for i in xrange(m): 
           #TODO : roots[0] is always 1. Does this make any sense? check javascript version
           if roots_rdc_u256[0] == t_u256:
             u_u256[i] = ZFieldElExt(1).reduce().as_uint256()
             return z, u_u256()

       l = z * ZFieldElExt(int(m)).inv().reduce()
       for i in xrange(m):
         x = t - ZFieldElRedc(BigInt.from_uint256(roots_rdc_u256[i]))
         x_inv = x.inv()
         u_u256[i] = (l * x_inv).as_uint256()
         l = l * omega

       return z, u_u256
   
    def _calculateEncryptedValuesAtT(self, toxic_t):
      a_t_u256, b_t_u256, c_t_u256, z_t = self._calculateValuesAtT(toxic_t)
      self.A  = np.zeros((self.nVars,NWORDS_256BIT),dtype=np.uint32)
      self.B1 = np.zeros((self.nVars,NWORDS_256BIT),dtype=np.uint32)
      self.B2 = np.zeros((self.nVars,NWORDS_256BIT),dtype=np.uint32)
      self.C  = np.zeros((self.nVars,NWORDS_256BIT),dtype=np.uint32)

      prime = ZField.get_extended_p()
      curve_params = self.curve_data['curve_params']
      curve_params_g2 = self.curve_data['curve_params_g2']

      toxic_kalfa = ZFieldElExt(randint(1,prime.as_long()-1))
      toxic_kbeta = ZFieldElExt(randint(1,prime.as_long()-1))
      toxic_kgamma = ZFieldElExt(randint(1,prime.as_long()-1))
      toxic_kdelta = ZFieldElExt(randint(1,prime.as_long()-1))

      toxic_invDelta = toxic_kdelta.inv()
      toxic_invGamma = toxic_kgamma.inv()

      ZPoly.init(GrothSetup.GroupIDX)
      Gx = ZFieldElExt(curve_params['Gx'])
      Gy = ZFieldElExt(curve_params['Gy'])
      G2x = Z2FieldEl([curve_params_g2['Gx1'], curve_params_g2['Gx2']])
      G2y = Z2FieldEl([curve_params_g2['Gy1'], curve_params_g2['Gy2']])

      self.vk_alfa_1 = ECCAffine([Gx,Gy]) * toxic_kalfa
      self.vk_beta_1 = ECCAffine([Gx,Gy]) * toxic_kbeta
      self.vk_delta_1 = ECCAffine([Gx,Gy]) * toxic_kdelta
      
      self.vk_beta_2 = ECCAffine([G2x, G2y]) * toxic_kbeta
      self.vk_delta_2 = ECCAffine([G2x, G2y]) * toxic_kdelta



    def _calculateValuesAtT(self, toxic_t):
       z_t, u = self._evalLagrangePoly(self.domainBits, toxic_t)

       #a_t_u256 = np.zeros((self.nVars, NWORDS_256BIT), dtype=np.uint32)
       #b_t_u256 = np.zeros((self.nVars, NWORDS_256BIT), dtype=np.uint32)
       #c_t_u256 = np.zeros((self.nVars, NWORDS_256BIT), dtype=np.uint32)

       #offsetA = self.polsA[0]+1
       #offsetB = self.polsB[0]+1
       #offsetC = self.polsC[0]+1
       pidx = ZField.get_field()

       a_t_u256 = mpoly_madd_h(self.polsA, u.reshape(-1), self.nVars, pidx)
       b_t_u256 = mpoly_madd_h(self.polsB, u.reshape(-1), self.nVars, pidx)
       c_t_u256 = mpoly_madd_h(self.polsC, u.reshape(-1), self.nVars, pidx)
       """
       for s in xrange(self.nVars):
         offsetA += self.polsA[s+1]
         a_t_u256[s] = mpoly_madd_h(self.polsA[offsetA:offsetA+self.polsA[s+1]*NWORDS_256BIT, u.reshape(-1), pidx)
         #for i in xrange(self.polsA[s+1]):
             #v = montmult_h(self.polsA[offsetA+i*NWORDS_256BIT:offsetA+(i+1)*NWORDS_256BIT],u[i],pidx)
             #a_t_u256[s] = addm_h(a_t_u256[s] , v, pidx)
 
         offsetB += self.polsB[s+1]
         for i in xrange(self.polsB[s+1]):
             v = montmult_h(self.polsB[offsetB+i*NWORDS_256BIT:offsetB+(i+1)*NWORDS_256BIT],u[i],pidx)
             b_t_u256[s] = addm_h(b_t_u256[s] , v, pidx)

         offsetC += self.polsC[s+1]
         for i in xrange(self.polsC[s+1]):
             v = montmult_h(self.polsC[offsetC+i*NWORDS_256BIT:offsetC+(i+1)*NWORDS_256BIT],u[i],pidx)
             c_t_u256[s] = addm_h(c_t_u256[s] , v, pidx)
         """


       return a_t_u256, b_t_u256, c_t_u256, z_t

    def _cirbin_to_u256(self, circuit_f):
        return readU256CircuitFile_h(circuit_f.encode("UTF-8"))

    def _ciru256_to_bin(self, ciru256_data, circuit_f):
        writeU256CircuitFile_h(ciru256_data, circuit_f.encode("UTF-8"))

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
        self.tformat       =  np.uint32(ciru256_data[CIRBIN_H_FORMAT_OFFSET])
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
                        self.tformat,
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
                  'tformat' : self.tformat,
                  'R1CSA_nWords' : self.R1CSA_nWords,
                  'R1CSB_nWords' : self.R1CSB_nWords,
                  'R1CSC_nWords' : self.R1CSC_nWords}

    def _cirjson_to_u256(self,circuit_f, in_format=ZUtils.FEXT, out_format=ZUtils.FEXT):
        """
          Converts from circom .json output file to binary format required to 
            calculate snarks setup. Only the following entries are used:
             - constraints -> R1CS a,b,c
             - nPubInputs  -> k
             - nVars       -> N
             - nOutputs    ->
             - tformat      -> EXT[0]/MONT[1]

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
            tformat : Extended[0]/Montgomery[1]----------------- 32 bits
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

        if in_format == out_format:
          R1CSA_u256 = [ZPolySparse(coeff[0]).as_uint256() for coeff in cir_data['constraints']]
        elif in_format == ZUtils.FEXT:
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
                

        if in_format == out_format:
          R1CSB_u256 = [ZPolySparse(coeff[1]).as_uint256() for coeff in cir_data['constraints']]
        elif in_format == ZUtils.FEXT:
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

        if in_format == out_format:
          R1CSC_u256 = [ZPolySparse(coeff[2]).as_uint256() for coeff in cir_data['constraints']]
        elif in_format == ZUtils.FEXT:
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
        self.tformat       =  np.uint32(out_format)
        self.R1CSA_nWords =  np.uint32(R1CSA_len)
        self.R1CSB_nWords =  np.uint32(R1CSB_len)
        self.R1CSC_nWords =  np.uint32(R1CSC_len)
        self.R1CSA        =  R1CSA_u256
        self.R1CSB        =  R1CSB_u256
        self.R1CSC        =  R1CSC_u256 

        return  self._cirvarsPack()


if __name__ == "__main__":
    in_circuit_f = '../../data/prove-kyc.json'
    out_circuit_f = '../../data/prove-kyc.bin'
    #in_circuit_f = '../../data/circuit.json'
    #out_circuit_f = '../../data/circuit.bin'
    #GS = GrothSetup(in_circuit_f=in_circuit_f, out_circuit_f=out_circuit_f, in_format=ZUtils.FEXT, out_format=ZUtils.FEXT)

    #in_circuit_f = '../../data/prove-kyc.bin'
    GS = GrothSetup(in_circuit_f=out_circuit_f)
    GS.setup()
