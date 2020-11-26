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

INPUT_R1CS_F = "test_circom_c.r1cs"
INPUT_PKBIN_F      = "test_circom_pk.bin"
INPUT_PKZKEY_F      = "test_circom_pk.zkey"
INPUT_HERMEZ_PKZKEY_F      = "hermez_pk.zkey"
INPUT_VK_F      = "test_circom_vk.json"
OUTPUT_P_F      = "test_circom_p.json"
OUTPUT_PD_F      = "test_circom_pd.json"
OUTPUT_HERMEZP_F      = "hermez_p.json"
OUTPUT_HERMEZPD_F      = "hermez_pd.json"


INPUT_CIRCUIT_F = "test_circom.cir"
INPUT_DATA_F = "test_circom_input.json"
WITNESS_WTNS_F = "test_circom_w.wtns"
WITNESS_WSHM_F = "test_circom_w.wshm"
WITNESS_HERMEZ_WTNS_F = "hermez_w.wtns"
WITNESS_HERMEZ_WSHM_F = "hermez_w.wshm"


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
      witness_wtns_f = circuits_folder+ WITNESS_WTNS_F
      witness_wshm_f = circuits_folder+ WITNESS_WSHM_F
      witness_hermez_wtns_f = circuits_folder+ WITNESS_HERMEZ_WTNS_F
      witness_hermez_wshm_f = circuits_folder+ WITNESS_HERMEZ_WSHM_F
  
      os.chdir(cusnarks_folder+"/aux_data")

      print("Compiling Circuit and runinning TS with Snarkjs...\n")

      call(["./build_zkey.sh", "12"])
      call(["./launch_circom.sh"])
  
      snarkjs_folder = cfg.get_snarkjs_folder()
 
      os.chdir(cusnarks_folder)

      pkfile_bin=INPUT_PKBIN_F
      pkfile_zkey=INPUT_PKZKEY_F
      pkfile_hermez_zkey=INPUT_HERMEZ_PKZKEY_F
      
      
      if os.path.exists(circuits_folder+pkfile_bin):
         os.remove(circuits_folder+pkfile_bin)
      if os.path.exists(circuits_folder+INPUT_VK_F):
         os.remove(circuits_folder+INPUT_VK_F)

      print("Launching Setup with Cusnarks.....\n")
      GS = GrothSetup(in_circuit_f = circuits_folder+INPUT_R1CS_F, out_pk_f=circuits_folder+pkfile_bin,
                      out_vk_f=circuits_folder+INPUT_VK_F, keep_f=circuits_folder, 
                      snarkjs=snarkjs_folder, seed=123)
      GS.setup()
      del GS
      
      if os.path.exists(circuits_folder+OUTPUT_PD_F):
         os.remove(circuits_folder+OUTPUT_PD_F)
      if os.path.exists(circuits_folder+OUTPUT_P_F):
         os.remove(circuits_folder+OUTPUT_P_F)
     
      GP = GrothProver(circuits_folder+pkfile_bin, verification_key_f = circuits_folder+INPUT_VK_F, 
                       start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=0, seed=123)
      result = GP.proof(witness_wtns_f, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)
      del GP

      test_printResults(result, "TS : Cusnarks R1CS.  Proof: PKBIN + WTNS. CPU" )

      if use_pycusnarks :
        if os.path.exists(circuits_folder+OUTPUT_PD_F):
           os.remove(circuits_folder+OUTPUT_PD_F)
        if os.path.exists(circuits_folder+OUTPUT_P_F):
           os.remove(circuits_folder+OUTPUT_P_F)
       
        GP = GrothProver(circuits_folder+pkfile_bin, verification_key_f = circuits_folder+INPUT_VK_F, 
                         start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=4, seed=123)
        result = GP.proof(witness_wtns_f, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)
        del GP
  
        test_printResults(result, "TS : Cusnarks R1CS.  Proof: PKBIN + WTNS. GPU" )
     
      if os.path.exists(circuits_folder+OUTPUT_PD_F):
         os.remove(circuits_folder+OUTPUT_PD_F)
      if os.path.exists(circuits_folder+OUTPUT_P_F):
         os.remove(circuits_folder+OUTPUT_P_F)
      GP = GrothProver(circuits_folder+pkfile_bin, verification_key_f = circuits_folder+INPUT_VK_F, 
                       start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=0, seed=123)
      result = GP.proof(witness_wshm_f, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)

      del GP
      
      test_printResults(result, "TS : Cusnarks R1CS.  Proof: PKBIN + WSHM. CPU")
      
      if use_pycusnarks:
        if os.path.exists(circuits_folder+OUTPUT_PD_F):
           os.remove(circuits_folder+OUTPUT_PD_F)
        if os.path.exists(circuits_folder+OUTPUT_P_F):
           os.remove(circuits_folder+OUTPUT_P_F)
        GP = GrothProver(circuits_folder+pkfile_bin, verification_key_f = circuits_folder+INPUT_VK_F, 
                         start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=4, seed=123)
        result = GP.proof(witness_wshm_f, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)

        del GP
      
        test_printResults(result, "TS : Cusnarks R1CS.  Proof: PKBIN + WSHM. GPU")

      if os.path.exists(circuits_folder+OUTPUT_PD_F):
         os.remove(circuits_folder+OUTPUT_PD_F)
      if os.path.exists(circuits_folder+OUTPUT_P_F):
         os.remove(circuits_folder+OUTPUT_P_F)
      
      GP = GrothProver(circuits_folder+pkfile_zkey, 
                       start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=0, seed=123)
      result = GP.proof(witness_wtns_f, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)

      del GP
      test_printResults(result, "TS : SNARKJS. Proof:  ZKEY + WTNS. CPU")
      
      if use_pycusnarks:
        if os.path.exists(circuits_folder+OUTPUT_PD_F):
           os.remove(circuits_folder+OUTPUT_PD_F)
        if os.path.exists(circuits_folder+OUTPUT_P_F):
           os.remove(circuits_folder+OUTPUT_P_F)
        
        GP = GrothProver(circuits_folder+pkfile_zkey, 
                         start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=4, seed=123)
        result = GP.proof(witness_wtns_f, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)

        del GP
        test_printResults(result, "TS : SNARKJS. Proof:  ZKEY + WTNS. GPU")
      
      if os.path.exists(circuits_folder+OUTPUT_PD_F):
         os.remove(circuits_folder+OUTPUT_PD_F)
      if os.path.exists(circuits_folder+OUTPUT_P_F):
         os.remove(circuits_folder+OUTPUT_P_F)
      GP = GrothProver(circuits_folder+pkfile_zkey, 
                       start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=0, seed=123)
      result = GP.proof(witness_wshm_f, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)

      del GP
      test_printResults(result, "TS : SNARKJS. Proof:  ZKEY + WSHM. CPU")

      if use_pycusnarks:
        if os.path.exists(circuits_folder+OUTPUT_PD_F):
           os.remove(circuits_folder+OUTPUT_PD_F)
        if os.path.exists(circuits_folder+OUTPUT_P_F):
           os.remove(circuits_folder+OUTPUT_P_F)
        GP = GrothProver(circuits_folder+pkfile_zkey, 
                         start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=4, seed=123)
        result = GP.proof(witness_wshm_f, circuits_folder+OUTPUT_P_F, circuits_folder+OUTPUT_PD_F, verify_en=1)
  
        del GP
        test_printResults(result, "TS : SNARKJS. Proof:  ZKEY + WSHM. GPU")

      os.chdir(cusnarks_folder+"/aux_data")

      os.chdir(cusnarks_folder+"/aux_data")
      call(["./launch_hermez.sh"])
      os.chdir(cusnarks_folder)

      if os.path.exists(circuits_folder+OUTPUT_HERMEZPD_F):
         os.remove(circuits_folder+OUTPUT_HERMEZPD_F)
      if os.path.exists(circuits_folder+OUTPUT_HERMEZP_F):
         os.remove(circuits_folder+OUTPUT_HERMEZP_F)
      
      GP = GrothProver(circuits_folder+pkfile_hermez_zkey, 
                       start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=0, seed=123)
      result = GP.proof(witness_hermez_wtns_f, circuits_folder+OUTPUT_HERMEZP_F, circuits_folder+OUTPUT_HERMEZPD_F, verify_en=1)

      del GP
      test_printResults(result, "TS : SNARKJS. Proof:  ZKEY(HERMEZ) + WTNS. CPU")

      if use_pycusnarks:
        if os.path.exists(circuits_folder+OUTPUT_HERMEZPD_F):
           os.remove(circuits_folder+OUTPUT_HERMEZPD_F)
        if os.path.exists(circuits_folder+OUTPUT_HERMEZP_F):
           os.remove(circuits_folder+OUTPUT_HERMEZP_F)
        
        GP = GrothProver(circuits_folder+pkfile_hermez_zkey, 
                         start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=4, seed=123)
        result = GP.proof(witness_hermez_wtns_f, circuits_folder+OUTPUT_HERMEZP_F, circuits_folder+OUTPUT_HERMEZPD_F, verify_en=1)
  
        del GP
        test_printResults(result, "TS : SNARKJS. Proof:  ZKEY(HERMEZ) + WTNS")
      
      
      if os.path.exists(circuits_folder+OUTPUT_HERMEZPD_F):
         os.remove(circuits_folder+OUTPUT_HERMEZPD_F)
      if os.path.exists(circuits_folder+OUTPUT_HERMEZP_F):
         os.remove(circuits_folder+OUTPUT_HERMEZP_F)
      GP = GrothProver(circuits_folder+pkfile_hermez_zkey, 
                       start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=0, seed=123)
      result = GP.proof(witness_hermez_wshm_f, circuits_folder+OUTPUT_HERMEZP_F, circuits_folder+OUTPUT_HERMEZPD_F, verify_en=1)

      del GP
      test_printResults(result, "TS : SNARKJS. Proof:  ZKEY(HERMEZ) + WSHM. CPU")

      if use_pycusnarks:
        if os.path.exists(circuits_folder+OUTPUT_HERMEZPD_F):
           os.remove(circuits_folder+OUTPUT_HERMEZPD_F)
        if os.path.exists(circuits_folder+OUTPUT_HERMEZP_F):
           os.remove(circuits_folder+OUTPUT_HERMEZP_F)
        GP = GrothProver(circuits_folder+pkfile_hermez_zkey, 
                         start_server=0, keep_f=circuits_folder, snarkjs=snarkjs_folder, n_gpus=4, seed=123)
        result = GP.proof(witness_hermez_wshm_f, circuits_folder+OUTPUT_HERMEZP_F, circuits_folder+OUTPUT_HERMEZPD_F, verify_en=1)
  
        del GP
        test_printResults(result, "TS : SNARKJS. Proof:  ZKEY(HERMEZ) + WSHM. GPU")

if __name__ == "__main__":
    result_pkbin = test_cusnarks()

