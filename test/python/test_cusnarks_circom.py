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
// File name  : test_circom.py
//
// Date       : 25/11/2019
//
// ------------------------------------------------------------------
//
// Description: 
//   Tests interoperability of cusnarks with circom and snarkjs.
// It requires both programs to be installed
//   
//
// TODO 
// ------------------------------------------------------------------

"""
import sys
from random import randint
import os

sys.path.append('../../src/python')
sys.path.append(os.path.abspath(os.path.dirname('../../config/')))

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
    from pycusnarks import *
    use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

from groth_prover import *
from groth_setup import *
from subprocess import call
import cusnarks_config as cfg
from termcolor import colored

INPUT_R1CS_F = "test_circom.cir.r1cs"
INPUT_PKBIN_F      = "test_circom_pk.bin"
INPUT_PKJSON_F      = "test_circom_pk.json"
INPUT_VK_F      = "test_circom_vk.json"
INPUT_W_F       = "test_circom_w.dat"
OUTPUT_P_F      = "test_circom_p.json"
OUTPUT_PD_F      = "test_circom_pd.json"


INPUT_CIRCUIT_F = "test_circom.cir"
INPUT_DATA_F = "test_circom_input.json"
WITNESS_F = "test_circom_w.json"


def test_printResults(result,ftype):
    if result == 1:
      print("Cusnarks ("+ftype+")  test PASSED\n")
    else:
      print(colored("Cusnarks ("+ftype+") test FAILED\n", 'red'))
      if not use_pycusnarks:
        print(colored("CUDA toolkit not available\n", 'red'))

    return

def test_cusnarks():
    if use_pycusnarks:
      circuits_folder = cfg.get_circuits_folder()
      cusnarks_folder = cfg.get_cusnarks_folder()+"test/python"
      in_circom_circuit_f = cusnarks_folder+"/aux_data/" + INPUT_CIRCUIT_F 
      out_circom_circuit_f = cusnarks_folder+"/aux_data/" + INPUT_CIRCUIT_F +".wasm"
      in_circom_data_f = cusnarks_folder+"/aux_data/" + INPUT_DATA_F
      witness_f = circuits_folder+ WITNESS_F
      constraints_f = in_circom_circuit_f+".r1cs"
  
      os.chdir(cusnarks_folder+"/aux_data")

      print("Compiling Circuit "+in_circom_circuit_f +" ---> "+out_circom_circuit_f+" ....\n")

      call(["circom", in_circom_circuit_f,"-r1cs", "--wasm", "--sym"])
  
      print("Generating Witness ---> "+ witness_f+" ....\n")
      call(["snarkjs", "calculatewitness", "--wasm", out_circom_circuit_f, "--input", in_circom_data_f, "--witness", witness_f])
      call(["mv", constraints_f, circuits_folder])
      os.remove(in_circom_circuit_f+".cpp")
      os.remove(in_circom_circuit_f+".wasm")
      os.remove(in_circom_circuit_f+".sym")

  
      snarkjs_folder = cfg.get_snarkjs_folder()
  
      os.chdir(cusnarks_folder)
      print("Launching Setup .....\n")
      pkfile=INPUT_PKBIN_F
      GS = GrothSetup(in_circuit_f = circuits_folder+INPUT_R1CS_F, out_pk_f=circuits_folder+pkfile,
                      out_vk_f=circuits_folder+INPUT_VK_F, keep_f=circuits_folder, 
                      snarkjs=snarkjs_folder, seed=123)
      GS.setup()
      del GS
  
      print("Launching Prover......\n")
      GP = GrothProver(circuits_folder+pkfile, verification_key_f = circuits_folder+INPUT_VK_F, 
                       start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=4, seed=123)
      result = GP.proof(witness_f, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)

      return result

if __name__ == "__main__":
    result_pkbin = test_cusnarks()
    test_printResults(result_pkbin, "PKBIN")

