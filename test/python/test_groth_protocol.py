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
// File name  : test_grooth_protocol.py
//
// Date       : 31/01/2019
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

from groth_protocol import *
from subprocess import call

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
    call(["rm", "-f", WITNESS_FILE,PROVINGKEY_FILE, VERIFICATION_FILE, PROOF_FILE, PUBLIC_FILE])
    call([SNARKJS, SETUP_COMMAND, CIRCUIT_FILE_OPT, CIRCUIT_FILE, PROVINGKEY_FILE_OPT, PROVINGKEY_FILE, VERIFICATION_FILE_OPT, VERIFICATION_FILE, PROTOCOL_OPT, PROTOCOL])
    call([SNARKJS, WITNESS_COMMAND, CIRCUIT_FILE_OPT, CIRCUIT_FILE, DATA_FILE_OPT, DATA_FILE, WITNESS_FILE_OPT, WITNESS_FILE])
    G = GrothSnarks(witness_f=WITNESS_FILE, proving_key_f=PROVINGKEY_FILE, proof_f=PROOF_FILE, public_f=PUBLIC_FILE)
    G.gen_proof()
    G.write_json()
    call([SNARKJS, VERIFY_COMMAND, VERIFICATION_FILE_OPT, VERIFICATION_FILE, PROOF_FILE_OPT, PROOF_FILE, PUBLIC_FILE_OPT, PUBLIC_FILE])

