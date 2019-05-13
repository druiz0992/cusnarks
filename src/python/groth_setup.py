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

class GrothSetupSnarks(object):
    
    GroupIDX = 0
    FieldIDX = 1

    def __init__(self, in_circuit_f=None, out_circuit_f=None, curve='BN128'):
  
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data['curve_params'])
        ZPoly.init(GrothSetupSnarks.FieldIDX)

        # cir Json to u256
        if in_circuit_f.endswith('.json'):
           cir_u256 = self.cirjson_to_u256(in_circuit_f)

           #u256 to bin
           if out_circuit_f is not None:
              self.ciru256_to_bin(cir_u256, out_circuit_f)

        elif in_circuit_f.endswith('.bin'):
             cir_u256 = self.cirbin_to_u256(in_circuit_f)
       
    def cirbin_to_u256(self, circuit_f):
        cir_u256 = readU256CircuitFile_h(circuit_f)
        return cir_u256

    def ciru256_to_bin(self, ciru256_data, circuit_f):
        writeU256CircuitFile_h(ciru256_data, circuit_f)

    def cirjson_to_u256(self,circuit_f):
        f = open(circuit_f,'r')
        cir_json_data = json.load(f)
        cir_data = json_to_dict(cir_json_data)
        f.close()

        pol_ncoeff = len(cir_data['constraints'])
        polsA_u256 = np.zeros(pol_ncoeff*(NWORDS_256BIT+1)+1, dtype=np.uint32)
        polsA_u256[0] = np.uint32(len(cir_data['constraints']))
        polsB_u256 = np.zeros(len(cir_data['constraints'])*(NWORDS_256BIT+1)+1, dtype=np.uint32)
        polsB_u256[0] = np.uint32(len(cir_data['constraints']))
        polsC_u256 = np.zeros(len(cir_data['constraints'])*(NWORDS_256BIT+1)+1, dtype=np.uint32)
        polsC_u256[0] = np.uint32(len(cir_data['constraints']))

        cidx = 1
        for c in cir_data['constraints']:
         
          polsA_u256[cidx] = np.uint32(c[0].keys()[0])
          polsA_u256[pol_ncoeff+1+(cidx-1)*NWORDS_256BIT:pol_ncoeff+1+cidx*NWORDS_256BIT] = BigInt(c[0].values()[0]).as_uint256()

          polsB_u256[cidx] = np.uint32(c[1].keys()[0])
          polsB_u256[pol_ncoeff+1+(cidx-1)*NWORDS_256BIT:pol_ncoeff+1+cidx*NWORDS_256BIT] = BigInt(c[1].values()[0]).as_uint256()

          polsC_u256[cidx] = np.uint32(c[2].keys()[0])
          polsC_u256[pol_ncoeff+1+(cidx-1)*NWORDS_256BIT:pol_ncoeff+1+cidx*NWORDS_256BIT] = BigInt(c[2].values()[0]).as_uint256()

          cidx += 1

        circuit_data = np.concatenate((
                       [np.uint32(cir_data['nPubInputs']),
                       np.uint32(cir_data['nOutputs']),
                       np.uint32(cir_data['nVars'])],
                       polsA_u256, polsB_u256, polsC_u256))

        return circuit_data

       
        

        


if __name__ == "__main__":
    in_circuit_f = '../../../factor/circuit.json'
    out_circuit_f = '../../data/circuit.bin'
    G = GrothSetupSnarks(in_circuit_f, out_circuit_f=out_circuit_f)
