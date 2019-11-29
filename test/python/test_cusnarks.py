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
// File name  : test_grooth_prover.py
//
// Date       : 25/11/2019
//
// ------------------------------------------------------------------
//
// Description:
//   
//
// TODO 
//    incorrect format  -> once asserts substituted by exceptions,
//         test incorrect formats can be done
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

INPUT_CIRCUIT_F = "test_cusnarks_c.bin"
INPUT_PK_F      = "test_cusnarks_pk.bin"
INPUT_VK_F      = "test_cusnarks_vk.json"
INPUT_W_F       = "test_cusnarks_w.dat"
OUTPUT_P_F      = "test_cusnarks_p.json"
OUTPUT_PD_F      = "test_cusnarks_pd.json"

RUST_COMMAND = "../../../target/release/za"
RUST_INPUT_F ="circuit.circom"
RUST_GEN_CONSTRAINTS_COMPILE_OPT = "compile"
RUST_GEN_CONSTRAINTS_CUDA_OPT = "--cuda="

RUST_GEN_WITNESS_OPT1 = "test"
RUST_GEN_WITNESS_OPT2 = "--skipcompile"
RUST_GEN_WITNESS_OPT3 = "--outputwitness"

OBJ_COMMAND = "objcopy"
OBJ_INPUT_OPT = "-I"
OBJ_INPUT_VAL = "binary"
OBJ_OUTPUT_OPT = "-O"
OBJ_OUTPUT_VAL = "binary"
OBJ_MODE_OPT = "--reverse-bytes=4"
OBJ_INPUT_F = "test1.binwitness"

PYTHON_TEST_INPUT_F = "test_cusnarks.circom"

if __name__ == "__main__":
    result = 0
    if use_pycusnarks:
      circuits_folder = cfg.get_circuits_folder()
      rust_folder = cfg.get_rust_folder() + "/interop/circuits/cuda"
  
      cusnarks_folder = cfg.get_cusnarks_folder()+"test/python"
      python_test_data_f = cusnarks_folder+"/aux_data/" + PYTHON_TEST_INPUT_F 
  
      os.chdir(rust_folder)
  
      call(["cp", "-f", python_test_data_f, RUST_INPUT_F])
      call(["rm", "-f", INPUT_CIRCUIT_F])
      call(["rm", "-f", INPUT_W_F])
      call(["rm", "-f", OBJ_INPUT_F])
  
      call([RUST_COMMAND, RUST_GEN_CONSTRAINTS_COMPILE_OPT, RUST_GEN_CONSTRAINTS_CUDA_OPT+INPUT_CIRCUIT_F])
      call([RUST_COMMAND, RUST_GEN_WITNESS_OPT1, RUST_GEN_WITNESS_OPT2, RUST_GEN_WITNESS_OPT3])
  
      call([OBJ_COMMAND, OBJ_INPUT_OPT, OBJ_INPUT_VAL, OBJ_OUTPUT_OPT, OBJ_OUTPUT_VAL, OBJ_MODE_OPT, OBJ_INPUT_F, INPUT_W_F])
  
      call(["mv", INPUT_CIRCUIT_F, circuits_folder])
      call(["mv", "-f", INPUT_W_F, circuits_folder])
  
      call(["rm", "-f", "*test_cusnarks*"])
  
      os.chdir(cusnarks_folder)
  
      snarkjs_folder = cfg.get_snarkjs_folder()
  
      # launch setup
      GS = GrothSetup(in_circuit_f = circuits_folder+INPUT_CIRCUIT_F, out_pk_f=circuits_folder+INPUT_PK_F,
                      out_vk_f=circuits_folder+INPUT_VK_F, keep_f=circuits_folder,
                      snarkjs=snarkjs_folder, seed=123)
      GS.setup()
      del GS
  
      GP = GrothProver(circuits_folder+INPUT_PK_F, verification_key_f = circuits_folder+INPUT_VK_F, 
                       start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=4, seed=123)
      result = GP.proof(circuits_folder+INPUT_W_F, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)

    if result == 1:
      print("Cusnarks test PASSED\n")
    else:
      print(colored("Cusnarks test FAILED\n", 'red'))
      if not use_pycusnarks:
        print(colored("CUDA toolkit not available\n", 'red'))
