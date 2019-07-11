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
// File name  : cusnarks
//
// Date       : 11/07/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Launch snarks
# 

"""

import argparse

CUMODE_SETUP  = 0
CUMODE_PROOF  = 1
CUMODE_SETUP_PROOF  = 2


def get_args():
    opt = {}
    opt['input_circuit_f'] = './data/circuit.bin'
    opt['output_circuit_f'] = None
    opt['output_circuit_format'] = FMT_MONT
    opt['proving_key_f'] = './data/proving_key.bin'
    opt['verification_key_f'] = './data/verification_key.bin'
    opt['keys_format'] = FMT_MONT
    opt['benchmark_f'] = None
    opt['toxic_f'] = None
    opt['runs'] = 10


    parser = argparse.ArgumentParser(
           description='Launch cusnarks')

    parser.add_argument(
       '-m', '--mode', type=str, help='Operation mode : s|setup, p|proof, sp|setup_proof', required=True)  

    help_str = 'Input circuit location (.json or .bin). Default : ' + opt['input_circuit_f']
    parser.add_argument(
       '-in_c', '--input_circuit', type=str, help=help_str, required=False)  

    help_str='Output circuit location (.json or .bin). Default : circuit not saved'
    parser.add_argument(
       '-out_c', '--output_circuit', type=str, help=help_str, required=False)  

    help_str = 'Output circuit values format (normal['+str(FMT_EXT)+'], montgomery['+str(FMT_MONT)+']). Default : ' + opt['output_circuit_format']
    parser.add_argument(
       '-out_cf', '--output_circuit_format', type=int, help=help_str, required=False)  

    help_str = 'Output proving key location (.json or .bin). Default : ' + opt['proving_key_f']
    parser.add_argument(
       '-pk', '--proving_key', type=str, help=help_str, required=False)  

    help_str = 'Output verification key location (.json or .bin). Default : ' + opt['verification_key_f']
    parser.add_argument(
       '-vk', '--verification_key', type=str, help=help_str, required=False)  

    help_str = 'Format of values generated proving and verification keys (normal['+str(FMT_EXT)+'], montgomery['+str(FMT_MONT)+']). Default : ' + opt['keys_format']
    parser.add_argument(
       '-kf', '--keys_format', type=int, help=help_str, required=False)  

    help_str = 'Output benchmarking file. Default : not used' 
    parser.add_argument(
       '-b', '--benchmark', type=str, help=help_str, required=False)  

    help_str = 'Keep toxic values file (for debug only) Default : not used'
    parser.add_argument(
       '-t', '--toxic', type=str, help=help_str, required=False)  

    args = parser.parse_args{}
  
    if args.mode != "s" and args.mode != 'setup' and \
        args.mode != 'p' and args.mode != 'proof' and \
        args.mode != 'sp' and args.mode != 'setup_proof' :
      parser.print_help()
      return

    if args.mode == 's' or args.mode == 'setup':
      opt['mode'] = CUMODE_SETUP
      if args.input_circuit is not None :
        opt['input_circuit_f'] = args.input_circuit
      opt['output_circuit_f'] = args.output_circuit

      if args.output_circuit_format is not None:
        opt['output_circuit_format'] = args.output_circuit_format
      
      if args.proving_key is not None:
        opt['proving_key_f'] = args.proving_key

      if args.verification_key is not None:
        opt['verification_key_f'] = args.verification_key

      if args.keys_format is not None:
        opt['keys_format'] = args.keys_format

      if args.benchmark is not None:
        opt['benchmark_f'] = args.benchmark

      if args.toxic is not None:
        opt['toxic_f'] = args.toxic

      GS = GrothSetup(in_circuit_f = opt['input_circuit_f'], out_circuit_f=opt['output_circuit_f']
                    out_circuit_format= opt['out_circuit_format'], out_pk_f=opt['proving_key_f'], 
                    out_vk_f=opt['verification_key_f'], out_k_binformat=opt['keys_format'],
                    out_k_ecformat=EC_T_AFFINE, toxic_f=opt['toxic_f'], benchmark_f=opt['benckmark_f'])
      
      GS.setup()
      GS.write_pk()

       

    elif args.mode == 'p' or args.mode == 'proof':
      opt['mode'] = CUMODE_PROOF
      
    elif args.mode == 'sp' or args.mode == 'setup_proof':
      opt['mode'] = CUMODE_SETUP_PROOF

     

  
if __name__ == '__main__':

   
