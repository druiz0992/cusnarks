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
from pysnarks_utils import *

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

class GrothSetup(object):
    
    GroupIDX = 0
    FieldIDX = 1

    def __init__(self, curve='BN128', in_circuit_f=None, out_circuit_f=None):
  
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data['curve_params'])
        ZPoly.init(GrothSetup.FieldIDX)

        self.vk_proof = {}
        self.vk_verifier = {}

        self.nWords       = None
        self.nPubInputs   = None
        self.nOutputs     = None
        self.nVars        = None
        self.nConstraints = None
        self.R1CSA_nWords = None
        self.R1CSB_nWords = None
        self.R1CSC_nWords = None
        self.R1CSA        = None
        self.R1CSB        = None
        self.R1CSC        = None
        self.header       = None

        if in_circuit_f is not None:
           self.circuitRead(in_circuit_f,out_circuit_f)

    def circuitRead(self,circuit_f, out_circuit_f=None):
        # cir Json to u256
        if in_circuit_f.endswith('.json'):
           cir_u256 = self._cirjson_to_u256(in_circuit_f)

           #u256 to bin
           if out_circuit_f is not None:
              self._ciru256_to_bin(cir_u256, out_circuit_f)

        elif in_circuit_f.endswith('.bin'):
             cir_u256 = self._cirbin_to_u256(in_circuit_f)

        self._ciru256_to_vars(cir_u256)

    def setup(self):
        ZPoly.init(GrothSetup.FieldIDX)
        domainBits =  np.uint32(math.ceil(math.log(self.nConstraints+ 
                                           self.nPubInputs + 
                                           self.nOutputs,2)))


        self.vk_proof = { 'nVars' : self.nVars,
                     'nPublic' : self.nPubInputs + self.nOutputs,
                     'domainBits' : domainBits,
                     'domainSize' : 1 << domainBits,
                     'R1CSA' : self.R1CSA,
                     'R1CSB' : self.R1CSB,
                     'R1CSC' : self.R1CSC}

        self.vk_verifier = {'nPublic' : self.nPubInputs + self.nOutputs}
  
        prime = ZField.get_extended_p()
        toxic = randint(0,prime.as_long()-1)

        self._calculatePoly()
        self.calculateEncryptedValuesAtT(cirvars,toxic)

        return 


    def _calculatePoly(self):
        self._computeHeader()

        while True:
           pout_len = self.header['R1CSA_nWords']/8
           ret_v, polsA = r1cs_to_zpoly_h(self.R1CSA, self.header, pout_len, 1)
           if ret_v == 1:
              break
           pout_len *= 2

        while True:
           pout_len = self.header['R1CSB_nWords']/8
           ret_v, polsB = r1cs_to_zpoly_h(self.R1CSB, self.header, pout_len, 0)
           if ret_v == 1:
              break
           pout_len *= 2

        while True:
           pout_len = self.header['R1CSC_nWords']/8
           ret_v, polsC = r1cs_to_zpoly_h(self.R1CSC, self.header, pout_len, 0)
           if ret_v == 1:
              break
           pout_len *= 2

    def calculateEncryptedValuesAtT(self, cirvars,toxic):
       return

    def _cirbin_to_u256(self, circuit_f):
        return readU256CircuitFile_h(circuit_f)

    def _ciru256_to_bin(self, ciru256_data, circuit_f):
        writeU256CircuitFile_h(ciru256_data, circuit_f)

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

        R1CSA_u256 = [ZPolySparse(coeff[0]).as_uint256() for coeff in cir_data['constraints']]
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
                

        R1CSB_u256 = [ZPolySparse(coeff[1]).as_uint256() for coeff in cir_data['constraints']]
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

        R1CSC_u256 = [ZPolySparse(coeff[2]).as_uint256() for coeff in cir_data['constraints']]
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
        self.R1CSA_nWords =  np.uint32(R1CSA_len)
        self.R1CSB_nWords =  np.uint32(R1CSB_len)
        self.R1CSC_nWords =  np.uint32(R1CSC_len)
        self.R1CSA        =  R1CSA_u256
        self.R1CSB        =  R1CSB_u256
        self.R1CSC        =  R1CSC_u256 

        return  self._cirvarsPack()


if __name__ == "__main__":
    #in_circuit_f = '../../data/prove-kyc.json'
    #out_circuit_f = '../../data/prove-kyc.bin'
    #GS = GrothSetup(in_circuit_f=in_circuit_f, out_circuit_f=out_circuit_f)

    in_circuit_f = '../../data/prove-kyc.bin'
    GS = GrothSetup(in_circuit_f=in_circuit_f)
    GS.setup()
