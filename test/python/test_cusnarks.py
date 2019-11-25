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

sys.path.append('../../src/python')
sys.path.append(os.path.abspath(os.path.dirname('../../config/')))

from groth_prover import *
from subprocess import call
import os
import cusnarks_config as cfg

RUST_COMMAND = "../../../target/release/za"
RUST_GEN_CONSTRAINTS_COMPILE_OPT = "compile"
RUST_GEN_CONSTRAINTS_CUDA_OPT = "--cuda=test_c.bin"

RUST_GEN_WITNESS_OPT1 = "test"
RUST_GEN_WITNESS_OPT2 = "--skipcompile"
RUST_GEN_WITNESS_OPT3 = "--outputwitness"

OBJ_COMMAND = "objcopy"
OBJ_INPUT_OPT = "-I"
OBJ_INPUT_VAL = "binary"
OBJ_OUTPUT_OPT = "-O"
OBJ_OUTPUT_VAL = "binary"
OBJ_MODE_OTP = "--reverse-bytes=4"
OBJ_INPUT_FILE = "test1.binwitness"
OBJ_OUTPUT_FILE = "test_w.dat"

SNARKJS ="snarkjs"
SETUP_COMMAND ="setup"
WITNESS_COMMAND="calculatewitness"
VERIFY_COMMAND = "verify"
PROTOCOL_OPT = "--protocol"
PROTOCOL = "groth"
CIRCUIT_FILE_OPT = "-c"
CIRCUIT_FILE = "./aux_data/circuit.json"
DATA_FILE_OPT = "-i"
DATA_FILE = "./aux_data/input.json"
WITNESS_FILE_OPT = "-w"
WITNESS_FILE = "./aux_data/witness.json"
PROVINGKEY_FILE_OPT = "--pk"
PROVINGKEY_FILE = "./aux_data/proving_key.json"
VERIFICATION_FILE_OPT = "--vk"
VERIFICATION_FILE = "./aux_data/verification_key.json"
PROOF_FILE_OPT = "-p"
PROOF_FILE = "./aux_data/proof.json"
PUBLIC_FILE_OPT = "--pub"
PUBLIC_FILE = "./aux_data/public.json"


if __name__ == "__main__":
    # cd rust_folder
    rust_folder = cfg.get_rust_folder() + "/interop/circuits/cuda"
    os.chdir(rust_folder)
    call([RUST_COMMAND, RUST_GEN_CONSTRAINTS_COMPILE, RUST_GEN_CONSTRAINTS_CUDA_OPT])
    call([RUST_COMMAND, RUST_GEN_WITNESS_OPT1, RUST_GEN_WITNESS_OPT2, RUST_GEN_WITNESS_OPT3])
    call([RUST_COMMAND, RUST_GEN_WITNESS_OPT1, RUST_GEN_WITNESS_OPT2, RUST_GEN_WITNESS_OPT3])

    call([OBJ_COMMAND, OBJ_INPUT_OPT, OBJ_INPUT_VAL, OBJ_OUTPUT_OPT, OBJ_OUTPUT_VAL, OBJ_MODE_OPT, OBJ_INPUT_FILE, OBJ_OUTPUT_FILE])


