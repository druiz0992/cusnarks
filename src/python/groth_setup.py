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

#Input files
DEFAULT_WITNESS_LOC = "../../data/witness_pedersen.json"
DEFAULT_PROVING_KEY_LOC = "../../data/proving_key_pedersen.json"
#DEFAULT_WITNESS_LOC = "../../data/witness_multiplier.json"
#DEFAULT_PROVING_KEY_LOC = "../../data/proving_key_multiplier.json"
#Output files
DEFAULT_PROOF_LOC =  "../../data/proof.json"
DEFAULT_PUBLIC_LOC =  "../../data/public.json"
DATA_FOLDER = "../../data/"

ROOTS_1M_filename = '../../data/zpoly_data_1M.npz'

class GrothSetup(object):
    
    GroupIDX = 0
    FieldIDX = 1

    def __init__(self, in_circuit_f, out_circuit_f=None, curve='BN128'):
  
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data['curve_params'])
        ZPoly.init(GrothSetup.FieldIDX)
        self.vk_proof = {}
        self.vk_verifier = {}

        # cir Json to u256
        if in_circuit_f.endswith('.json'):
           cir_u256 = self.cirjson_to_u256(in_circuit_f)

           #u256 to bin
           if out_circuit_f is not None:
              self.ciru256_to_bin(cir_u256, out_circuit_f)

        elif in_circuit_f.endswith('.bin'):
             cir_u256 = self.cirbin_to_u256(in_circuit_f)

        cir_vars = self.ciru256_to_vars(cir_u256)

        self.setup(cir_vars)

    def setup(self,cirvars):
        ZPoly.init(GrothSetup.FieldIDX)
        domainBits =  np.uint32(math.ceil(math.log(cirvars['nConstraints']+ 
                                           cirvars['nPubInputs'] + 
                                           cirvars['nOutputs'],2)))
        self.vk_proof = { 'nVars' : cirvars['nVars'],
                     'nPublic' : cirvars['nPubInputs'] + cirvars['nOutputs'],
                     'domainBits' : domainBits,
                     'domainSize' : 1 << domainBits,
                     'constA' : cirvars['constA'],
                     'constB' : cirvars['constB'],
                     'constC' : cirvars['constC']}

        self.vk_verifier = {'nPublic' : cirvars['nPubInputs'] + cirvars['nOutputs']}
  
        prime = ZField.get_extended_p()
        toxic = randint(0,prime.as_long()-1)

        self.calculatePoly(cirvars)
        self.calculateEncryptedValuesAtT(cirvars,toxic)

        return 

    def calculatePoly(self, cirvars):
        polsA = constraints_to_poly(cirvars['constA'])
        polsB = constraints_to_poly(cirvars['constB'])
        polsC = constraints_to_poly(cirvars['constC'])

    def calculateEncryptedValuesAtT(self, cirvars,toxic):
       return

    def cirbin_to_u256(self, circuit_f):
        cir_u256 = readU256CircuitFile_h(circuit_f)
        return cir_u256

    def ciru256_to_bin(self, ciru256_data, circuit_f):
        writeU256CircuitFile_h(ciru256_data, circuit_f)

    def ciru256_to_vars(self, ciru256_data):
        constA_offset = CIRBIN_H_N_OFFSET
        constB_offset = CIRBIN_H_N_OFFSET +  \
                       np.uint32(ciru256_data[CIRBIN_H_CONSTA_NWORDS_OFFSET])
        constC_offset = CIRBIN_H_N_OFFSET + \
                       np.uint32(ciru256_data[CIRBIN_H_CONSTA_NWORDS_OFFSET]) + \
                       np.uint32(ciru256_data[CIRBIN_H_CONSTB_NWORDS_OFFSET])

        cirvars = {'nWords'     : np.uint32(ciru256_data[CIRBIN_H_NWORDS_OFFSET]),
                    'nPubInputs' : np.uint32(ciru256_data[CIRBIN_H_NPUBINPUTS_OFFSET]),
                    'nOutputs'   : np.uint32(ciru256_data[CIRBIN_H_NOUTPUTS_OFFSET]),
                    'nVars'      : np.uint32(ciru256_data[CIRBIN_H_NVARS_OFFSET]),
                    'nConstraints' : np.uint32(ciru256_data[CIRBIN_H_NCONSTRAINTS_OFFSET]),
                    'constA_nWords' : np.uint32(ciru256_data[CIRBIN_H_CONSTA_NWORDS_OFFSET]),
                    'constB_nWords' : np.uint32(ciru256_data[CIRBIN_H_CONSTB_NWORDS_OFFSET]),
                    'constC_nWords' : np.uint32(ciru256_data[CIRBIN_H_CONSTC_NWORDS_OFFSET]),
                    'constA' : ciru256_data[constA_offset:constB_offset] ,
                    'constB' : ciru256_data[constB_offset:constC_offset],
                    'constC' : ciru256_data[constC_offset:] }

        return cirvars

    def cirvars_to_u256(self, cirvars):
        circuit_u256 = np.concatenate((
                       [cirvars['nWords'],
                        cirvars['nPubInputs'],
                        cirvars['nOutputs'],
                        cirvars['nVars'],
                        cirvars['nConstraints'],
                        cirvars['constA_nWords'],
                        cirvars['constB_nWords'],
                        cirvars['constC_nWords']],
                        cirvars['constA'],
                        cirvars['constB'],
                        cirvars['constC']))

        return circuit_u256

    def cirjson_to_u256(self,circuit_f):
        f = open(circuit_f,'r')
        cir_json_data = json.load(f)
        cir_data = json_to_dict(cir_json_data)
        f.close()

        constA_u256 = [ZPolySparse(coeff[0]).as_uint256() for coeff in cir_data['constraints']]
        constA_l = []
        constA_p = []
        for l,p in constA_u256:
            constA_l.append(l)
            constA_p.append(p)
        constA_u256 = np.asarray(np.concatenate((np.asarray([len(constA_l)]),
                                              np.concatenate(
                                                 (np.cumsum(constA_l), 
                                                  np.concatenate(constA_p))))),
                                                  dtype=np.uint32)

        constB_u256 = [ZPolySparse(coeff[1]).as_uint256() for coeff in cir_data['constraints']]
        constB_l = []
        constB_p = []
        for l,p in constB_u256:
            constB_l.append(l)
            constB_p.append(p)
        constB_u256 = np.asarray(np.concatenate((np.asarray([len(constB_l)]),
                                              np.concatenate(
                                                 (np.cumsum(constB_l), 
                                                  np.concatenate(constB_p))))),
                                                  dtype=np.uint32)

        constC_u256 = [ZPolySparse(coeff[2]).as_uint256() for coeff in cir_data['constraints']]
        constC_l = []
        constC_p = []
        for l,p in constC_u256:
            constC_l.append(l)
            constC_p.append(p)
        constC_u256 = np.asarray(np.concatenate((np.asarray([len(constC_l)]),
                                              np.concatenate(
                                                 (np.cumsum(constC_l), 
                                                  np.concatenate(constC_p))))),
                                                  dtype=np.uint32)

        pol_ncoeff =  np.uint32(cir_data['nVars'])

        fsize = CIRBIN_H_N_OFFSET + constA_u256.shape[0] + constB_u256.shape[0] + constC_u256.shape[0]
        cirvars = {'nWords'     : np.uint32(fsize),
                    'nPubInputs' : np.uint32(cir_data['nPubInputs']),
                    'nOutputs'   : np.uint32(cir_data['nOutputs']),
                    'nVars'      : np.uint32(cir_data['nVars']),
                    'nConstraints'  : np.uint32(len(cir_data['constraints'])),
                    'constA_nWords' : np.uint32(constA_u256.shape[0]),
                    'constB_nWords' : np.uint32(constB_u256.shape[0]),
                    'constC_nWords' : np.uint32(constC_u256.shape[0]),
                    'constA' :   constA_u256,
                    'constB' : constB_u256,
                    'constC' : constC_u256 }

        cir_u256data =  self.cirvars_to_u256(cirvars)

        return cir_u256data


if __name__ == "__main__":
    in_circuit_f = '../../data/prove-kyc.json'
    out_circuit_f = '../../data/prove-kyc.bin'
    G = GrothSetup(in_circuit_f, out_circuit_f=out_circuit_f)

    #in_circuit_f = '../../data/prove-kyc.bin'
    #G = GrothSetup(in_circuit_f)
