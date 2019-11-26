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

from groth_prover import *
from groth_setup import *
from subprocess import call
import cusnarks_config as cfg
from termcolor import colored

INPUT_CIRCUIT_F = "test_cusnarks_c.bin"
INPUT_PK_F      = "test_cusnarks_pk.bin"
INPUT_VK_F      = "test_cusnarks_vk.bin"
INPUT_W_F       = "test_cusnarks_w.dat"
OUTPUT_P_F      = "test_cusnarks_p.json"
OUTPU_PD_F      = "test_cusnarks_pd.json"

RUST_COMMAND = "../../../target/release/za"
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


if __name__ == "__main__":
    circuits_folder = cfg.get_circuits_folder()
    rust_folder = cfg.get_rust_folder() + "/interop/circuits/cuda"
    os.chdir(rust_folder)

    call(["rm", "-f", INPUT_CIRCUIT_F])
    call(["rm", "-f", INPUT_W_F])
    call(["rm", "-f", OBJ_INPUT_F])

    call([RUST_COMMAND, RUST_GEN_CONSTRAINTS_COMPILE_OPT, RUST_GEN_CONSTRAINTS_CUDA_OPT+INPUT_CIRCUIT_F])
    call([RUST_COMMAND, RUST_GEN_WITNESS_OPT1, RUST_GEN_WITNESS_OPT2, RUST_GEN_WITNESS_OPT3])

    call([OBJ_COMMAND, OBJ_INPUT_OPT, OBJ_INPUT_VAL, OBJ_OUTPUT_OPT, OBJ_OUTPUT_VAL, OBJ_MODE_OPT, OBJ_INPUT_F, INPUT_W_F])

    call(["mv", INPUT_CIRCUIT_F, circuits_folder])
    call(["mv", "-f", INPUT_W_F, circuits_folder])

    call(["rm", "-f", "*test_cusnarks*"])

    # launch setup
    GS = GrothSetup(in_circuit_f = INPUT_CIRCUIT_F, out_pk_f=INPUT_PK_F, out_vk_f=INPUT_VK_F, keep_f=circuits_folder)
    GS.setup()

    GP = GrothProver(INPUT_PK_F, INPUT_VK_F, verification_key_f=opt['verification_key_f'], start_server=0)
    result = GP.proof(INPUT_W_F, OUTPUT_P_F, OUTPU_PD_F, verify_en=1)

    if result == 1:
      print("Cusnarks test PASSED\n")
    else:
      print(colored("Cusnarks test FAILED\n", 'red'))
