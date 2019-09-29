#!/usr/bin/python3
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
// File name  : pysnarks
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
import os
import sys
import time
from subprocess import call

from groth_prover import *
from groth_setup import *
from pysnarks_utils import *

CUMODE_SETUP  = 0
CUMODE_PROOF  = 1

GP = None

sys.path.append(os.path.abspath(os.path.dirname('../../config/')))

import cusnarks_config as cfg

def init():
    opt = {}
    opt['data_f'] = cfg.get_circuits_folder()
    opt['input_circuit_f'] = 'circuit.bin'
    opt['verification_key_f'] = 'verification_key.json'
    opt['debug_f'] = 'toxic.json'
    opt['witness_f'] = 'witness.json'
    opt['proof_f'] = 'proof.json'
    opt['public_data_f']= 'public.json'

    opt['output_circuit_f'] = None
    opt['output_circuit_format'] = FMT_MONT
    opt['proving_key_f'] = 'proving_key.bin'
    opt['keys_format'] = FMT_MONT
    opt['charmander_f'] = '../../../charmander-circuit'
    opt['benchmark_f'] = '../../../charmander-circuit/test/compiled_circuits'
    opt['minL'] = 3
    opt['maxL'] = 10
    opt['nruns'] = 10
    opt['seed'] = None
    opt['snarkjs'] = cfg.get_snarkjs_folder()
    opt['debug'] = 0
    opt['witness_format'] = FMT_EXT
    opt['out_proving_key_format'] = FMT_MONT
    opt['out_proving_key_f'] = None
    opt['verify'] = 0
    opt['batch_size'] = 20
    opt['max_gpus'] = min(get_ngpu(max_used_percent=95.),2)
    opt['max_streams'] = get_nstreams()


    parser = argparse.ArgumentParser(
           description='Launch pysnarks')

    parser.add_argument(
       '-m', '--mode', type=str, help='Operation mode : s|setup, p|proof', required=True)  

    help_str = 'Input circuit location (.json or .bin). Default : ' + opt['input_circuit_f']
    parser.add_argument(
       '-in_c', '--input_circuit', type=str, help=help_str, required=False)  

    help_str='Output circuit location (.json or .bin). Default : circuit not saved'
    parser.add_argument(
       '-out_c', '--output_circuit', type=str, help=help_str, required=False)  

    help_str = 'Output circuit values format (normal['+str(FMT_EXT)+'], montgomery['+str(FMT_MONT)+']). Default : ' + str(opt['output_circuit_format'])
    parser.add_argument(
       '-out_cf', '--output_circuit_format', type=int, help=help_str, required=False)  

    help_str = 'Proving key location (.json or .bin). Default : ' + opt['proving_key_f']
    parser.add_argument(
       '-pk', '--proving_key', type=str, help=help_str, required=False)  

    help_str = 'Verification key location (.json or .bin). Default : ' + opt['verification_key_f']
    parser.add_argument(
       '-vk', '--verification_key', type=str, help=help_str, required=False)  

    help_str = 'Format of values generated proving and verification keys (normal['+str(FMT_EXT)+'], montgomery['+str(FMT_MONT)+']). Default : ' + str(opt['keys_format'])
    parser.add_argument(
       '-kf', '--keys_format', type=int, help=help_str, required=False)  

    help_str = 'Output benchmarking file. Default : not used' 
    parser.add_argument(
       '-b', '--benchmark', type=str, help=help_str, required=False)  

    help_str = 'Enables debug mode. It will launch snarkjs with known toxic values and will compare output file (Enable:1, Disable:0). Default : ' +str(opt['debug'])
    parser.add_argument(
       '-d', '--debug', type=int, help=help_str, required=False)  

    help_str = 'Sets seed for random number generator : Default : not set'
    parser.add_argument(
       '-seed', '--seed', type=int, help=help_str, required=False)  

    help_str = 'Sets location of snarkjs. Default : '+ opt['snarkjs']
    parser.add_argument(
       '-snarkjs', '--snarkjs', type=str, help=help_str, required=False)  

    help_str = 'Input Witness file. Default : ' + opt['witness_f']
    parser.add_argument(
       '-w', '--witness', type=str, help=help_str, required=False)  
  
    help_str = 'Input Witness file format (normal['+str(FMT_EXT)+'], montgomery['+str(FMT_MONT)+']). Default : ' + str(opt['witness_format'])
    parser.add_argument(
       '-wf', '--witness_format', type=int, help=help_str, required=False)  

    help_str = 'Output Proof file. Default : ' + opt['proof_f']
    parser.add_argument(
       '-p', '--proof', type=str, help=help_str, required=False)  
  
    help_str = 'Output Proof Public data file. Default : ' + opt['public_data_f']
    parser.add_argument(
       '-pd', '--public_data', type=str, help=help_str, required=False)  
  
    help_str = 'Mininum number of levels for benchmark testing. Default : ' +str(opt['minL'])
    parser.add_argument(
       '-ml', '--min_levels', type=int, help=help_str, required=False)  

    help_str = 'Maximum number of levels for benchmark testing. Default : ' +str(opt['maxL'])
    parser.add_argument(
       '-Ml', '--max_levels', type=int, help=help_str, required=False)  

    help_str = 'Number of repetitions for benchmark testing. Default : ' +str(opt['nruns'])
    parser.add_argument(
       '-rep', '--n_rep', type=int, help=help_str, required=False)  

    help_str = 'Output proving key file. Default : ' +str(opt['out_proving_key_f'])
    parser.add_argument(
       '-out_pk', '--out_proving_key', type=str, help=help_str, required=False)  

    help_str = 'Output proving key format (normal['+str(FMT_EXT)+'], montgomery['+str(FMT_MONT)+ ']). Default : ' +str(opt['out_proving_key_format'])
    parser.add_argument(
       '-out_kf', '--out_proving_key_format', type=int, help=help_str, required=False)  

    help_str = 'Run snarkjs proof verification. Default : ' + str(opt['verify'])
    parser.add_argument(
       '-v', '--verify', type=int, help=help_str, required=False)  

    help_str = 'Default location to retrieve and write files. Default ' + opt['data_f']
    parser.add_argument(
       '-df', '--data_folder', type=str, help=help_str, required=False)  

    help_str = 'Batch size (1 << batch_size). Default ' + str(opt['batch_size'])
    parser.add_argument(
       '-bs', '--batch_size', type=int, help=help_str, required=False)  

    help_str = 'Maximum nuumber of GPUs limit. Default ' + str(opt['max_gpus'])
    parser.add_argument(
       '-gpu', '--max_gpus', type=int, help=help_str, required=False)  

    help_str = 'Maximum nuumber of Streans limit. Default ' + str(opt['max_streams'])
    parser.add_argument(
       '-stream', '--max_streams', type=int, help=help_str, required=False)  


    return opt, parser

def run(opt, parser):
    args = parser.parse_args()

    if args.min_levels is not None:
       opt['minL'] = args.min_levels
 
    if args.max_levels is not None:
       opt['maxL'] = args.max_levels

    if args.n_rep is not None:
       opt['nruns'] = args.n_rep

    if args.benchmark is not None:
        opt['benchmark_f'] = args.benchmark

    if args.batch_size is not None:
       opt['batch_size'] = args.batch_size
    
    if args.max_gpus is not None:
       opt['max_gpus'] = args.max_gpus
    
    if args.max_streams is not None:
       opt['max_streams'] = args.max_streams
    

    if args.data_folder is not None:
        opt['data_f'] = args.data_f
        if not opt['data_f'].endswith('\\'):
           opt['data_f'] = opt['data_f'] + '\\'
    if not os.path.exists(opt['data_f'][:-1]):
        os.makedirs(opt['data_f'][:-1])
    opt['keep_f'] = opt['data_f']
        
    if args.mode != "s" and args.mode != 'setup' and \
        args.mode != 'p' and args.mode != 'proof' :
      parser.print_help()
      

    if args.proving_key is not None:
        if '/' in args.proving_key :
            opt['proving_key_f'] = args.proving_key
        else:
            opt['proving_key_f'] = opt['data_f']  + args.proving_key

    if args.verification_key is not None:
        if '/' in args.verification_key:
           opt['verification_key_f'] = args.verification_key
        else:
           opt['verification_key_f'] = opt['data_f'] + args.verification_key

    if args.seed is not None:
         opt['seed'] = args.seed

    if args.snarkjs is not None:
         opt['snarkjs'] = args.snarkjs

    if args.debug is 0 or args.debug is None:
         opt['debug_f'] = None
    else:
         opt['debug_f'] = opt['debug_f']


    if args.mode == 's' or args.mode == 'setup':

      opt['mode'] = CUMODE_SETUP

      if args.input_circuit is not None :
        if '/' in args.input_circuit:
           opt['input_circuit_f'] = args.input_circuit
        else:
           opt['input_circuit_f'] = opt['data_f'] + args.input_circuit

      if args.output_circuit is not None:
        if '/' in args.output_circuit:
           opt['output_circuit_f'] = args.output_circuit
        else:
           opt['output_circuit_f'] = opt['data_f'] + args.output_circuit

      if args.output_circuit_format is not None:
        opt['output_circuit_format'] = args.output_circuit_format
      
      if args.keys_format is not None:
        opt['keys_format'] = args.keys_format

      if args.benchmark is not None:
        # Compile circuit if it doesnt exist
        for level in xrange(opt['minL'], opt['maxL']+1):
           circuit = opt['benchmark_f'] + 'test-prove-kyc-'+str(level)+'-'+str(level)+'-'+str(level)+'.json'
           if not os.path.isfile(circuit):
              call([node, opt['charmander_f']+'inoutsZk/gen-inputs.js', level, level, level])
              call([node, opt['charmander_f']+'test/compile-circuit.js', level, level, level])
              


      GS = GrothSetup(in_circuit_f = opt['input_circuit_f'], out_circuit_f=opt['output_circuit_f'],
                    out_circuit_format= opt['output_circuit_format'], out_pk_f=opt['proving_key_f'], 
                    out_vk_f=opt['verification_key_f'], out_k_binformat=opt['keys_format'],
                    out_k_ecformat=EC_T_AFFINE, test_f=opt['debug_f'], benchmark_f=opt['benchmark_f'], seed=opt['seed'],
                    snarkjs=opt['snarkjs'], keep_f=opt['keep_f'], batch_size=opt['batch_size'])
      
      GS.setup()

    if args.mode == 'p' or args.mode == 'proof':
      opt['mode'] = CUMODE_PROOF

      if args.witness is not None:
        if '/' in args.witness:
           opt['witness_f'] = args.witness
        else:
           opt['witness_f'] = opt['data_f'] + args.witness

      if args.witness_format is not None:
          opt['witness_format'] = args.witness_format

      if args.proof is not None:
          if '/' in args.proof:
            opt['proof_f'] = args.proof
          else:
            opt['proof_f'] = opt['data_f'] + args.proof
   
      if args.public_data is not None:
          if '/' in args.public_data:
             opt['public_data_f'] = args.public_data
          else :
             opt['public_data_f'] = opt['data_f'] + args.public_data

      if args.verify is not None:
         opt['verify'] = args.verify

      if args.out_proving_key is not None:
         if '/' in args.out_proving_key:
            opt['out_proving_key_f'] = args.out_proving_key
         else:
            opt['out_proving_key_f'] = opt['data_f'] + args.out_proving_key

      if args.out_proving_key_format is not None:
         opt['out_proving_key_format'] = args.out_proving_key_format
    
      start = time.time()

      GP = GrothProver(opt['proving_key_f'], verification_key_f=opt['verification_key_f'], out_pk_f = opt['out_proving_key_f'],
                      out_pk_format = opt['out_proving_key_format'], test_f=opt['debug_f'],
                      benchmark_f=None, seed=opt['seed'], snarkjs=opt['snarkjs'], keep_f=opt['keep_f'])
      end = time.time() - start
      print("GP init : "+str(end))

      GP.proof(opt['witness_f'], opt['proof_f'], opt['public_data_f'],
              batch_size=opt['batch_size'], verify_en=opt['verify'], n_gpus=opt['max_gpus'], n_streams=opt['max_streams'] )
      
  
if __name__ == '__main__':

   opt, parser = init()

   run(opt, parser)

   """
   while True:
      run(opt, parser)

      sys.stdout.write('Enter new witness file: ')
      input_witness= sys.stdin.readline().rstrip()


      input_witness = input()
      opt['witness_f'] = opt['data_f'] + input_witness
      GP.proof(opt['witness_f'])
   """

