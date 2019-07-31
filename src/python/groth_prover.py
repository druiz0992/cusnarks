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
// File name  : groth_prover
//
// Date       : 27/01/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Zero Kowledge Groth prover implementation
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
from subprocess import call
import logging
from multiprocessing import RawArray
from ctypes import c_uint32

from zutils import ZUtils
import random
from zfield import *
from ecc import *
from zpoly import *
from constants import *
from pysnarks_utils import *
import multiprocessing as mp
from cuda_wrapper import *

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

sys.path.append(os.path.abspath(os.path.dirname('../../config/')))

import cusnarks_config as cfg

class GrothProver(object):
    
    def __init__(self, proving_key_f, verification_key_f=None,curve='BN128',  out_proof_f = None,
                 out_public_f = None, out_pk_f=None, out_pk_format=FMT_MONT,
                 out_proof_format= FMT_EXT, out_public_format=FMT_EXT, test_f=None, 
                 benchmark_f=None, seed=None, snarkjs=None, verify_en=0, keep_f=None):

        # Check valid folder exists
        if keep_f is None:
            print ("Repo directory needs to be provided\n")
            sys.exit(1)

        self.keep_f = gen_reponame(keep_f, sufix="_PROVER")

        logging.basicConfig(filename=self.keep_f + '/log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')

        if not use_pycusnarks :
          logging.error('PyCUSnarks shared library not found. Exiting...')
          sys.exit(1)

        if seed is not None:
          self.seed = seed
        else:
          x = os.urandom(4)
          self.seed = int(x.hex(),16)

        random.seed(self.seed) 

        self.roots_f = cfg.get_roots_file()
        self.n_bits_roots = cfg.get_n_roots()
        self.roots_rdc_u256_sh = RawArray(c_uint32,  (1 << self.n_bits_roots) * NWORDS_256BIT)
        self.roots_rdc_u256 = np.frombuffer(self.roots_rdc_u256_sh, dtype=np.uint32).reshape((1<<self.n_bits_roots, NWORDS_256BIT))
        np.copyto(self.roots_rdc_u256, readU256DataFile_h(self.roots_f.encode("UTF-8"), 1<<self.n_bits_roots, 1<<self.n_bits_roots) )

        batch_size = 1<<20
        self.ecbn128 = ECBN128(batch_size + 2,   seed=self.seed)
        self.ec2bn128 = EC2BN128(batch_size + 2, seed=self.seed)
        self.cuzpoly = ZCUPoly(batch_size, seed=self.seed)
    
        self.out_proving_key_f = out_pk_f
        self.out_proving_key_format = out_pk_format
        self.proving_key_f = proving_key_f
        self.verification_key_f = verification_key_f
        self.out_proof_f = out_proof_f
        self.out_public_f = out_public_f
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data)
        ZPoly.init(MOD_FIELD)

        ZField.set_field(MOD_FIELD)

        self.pk = getPK()

        self.pi_a_eccf1 = None
        self.pi_b_eccf2 = None
        self.pi_c_eccf1 = None
        self.pib1_eccf1 = None
        self.public_signals = None
        self.witness_f = None
        self.snarkjs = snarkjs
        self.verify_en = verify_en

        self.t_GP = {}
        self.t_EC = {}
        self.t_P = {}

        init_h()
        
        #scalar array : extended witness / polH
        self.scl_array = None
        self.sorted_scl_array = None
        self.sorted_scl_array_idx = None


        copy_input_files([proving_key_f, verification_key_f], self.keep_f)
        if test_f is None:
           self.test_f = test_f
        else:
           self.test_f= self.keep_f + '/' + test_f

        logging.info('#################################### ')
        logging.info('Staring Groth prover with the follwing parameters :')
        logging.info(' - curve : %s',curve)
        logging.info(' - proving_key_f : %s', proving_key_f)
        logging.info(' - verification_key_f : %s',verification_key_f)
        logging.info(' - out_proof_f : %s',out_proof_f)
        logging.info(' - out_pk_f : %s',out_pk_f)
        logging.info(' - out_pk_format : %s',out_pk_format) 
        logging.info(' - out_proof_format : %s',out_proof_format)
        logging.info(' - out_public_format :  %s',out_public_format)
        logging.info(' - test_f : %s',self.test_f)
        logging.info(' - benchmark_f : %s', benchmark_f)
        logging.info(' - seed : %s', self.seed)
        logging.info(' - snarkjs : %s', snarkjs)
        logging.info(' - verify_en : %s', verify_en)
        logging.info(' - keep_f : %s', keep_f)
        logging.info('#################################### ')
  
        self.load_pkdata()

        if self.out_proving_key_f is not None:
             if self.out_proving_key_f.endswith('.json'):
               pk_dict =pkvars_to_json(self.out_proving_key_format, EC_T_AFFINE, self.pk)
               pk_json = json.dumps(pk_dict, indent=4, sort_keys=True)
               f = open(self.out_proving_key_f, 'w')
               print(pk_json, file=f)
               f.close()
               del pk_dict
               del pk_json
             elif self.out_proving_key_f.endswith('.bin'):
               pk_bin = pkvars_to_bin(self.out_proving_key_format, EC_T_AFFINE, self.pk, ext=False)
               writeU256DataFile_h(pk_bin, self.out_proving_key_f.encode("UTF-8"))
               del pk_bin
             elif self.out_proving_key_f.endswith('.npz'):
                np.savez_compressed(proving_key_fnpz, alfa_1_u256 =  self.pk['alfa_1'],
                             beta_1_u256 = self.pk['beta_1'], delta_1_u256 = self.pk['delta_1'],
                             beta_2_u256 = self.pk['beta_2'], delta_2_u256 = self.pk['delta_2'],
                             A_u256 = self.pk['A'], B1_u256=self.pk['B1'], B2_u256 = self.pk['B2'],
                             C_u256 = self.pk['C'], hExps_u256 =self.pk['hExps'],
                             polsA_u256 = self.pk['polsA'], polsB_u256 = self.pk['polsB'], polsC_u256 = self.pk['polsC'],
                             nvars = self.pk['nVars'], npublic=self.pk['nPublic'], domain_bits=self.pk['domainBits'],
                             domain_size = self.pk['domainSize'])

        # convert data to array of bytes so that it can be easily transfered to shared mem
        if self.test_f :
           # if snarkjs is to be launched to compare results, I am assuming circuit is small, so i keep 
           # a version to be able to generate json. Else, results will overwrite input data
           self.pk_short = pkvars_to_bin(FMT_MONT, EC_T_AFFINE, self.pk, ext=False)

        pk_bin = pkvars_to_bin(FMT_MONT, EC_T_AFFINE, self.pk, ext=True)
        self.pk_sh = RawArray(c_uint32, int(pk_bin[0]))
        self.pk = np.frombuffer(self.pk_sh, dtype=np.uint32)
        np.copyto(self.pk, pk_bin)
        del pk_bin
             

    def __del__(self):
       release_h()

    def read_witness_data(self):
       ## Open and parse witness data
       if os.path.isfile(self.witness_f):

           pkbin_vars = pkbin_get(self.pk,['nVars','domainSize'])
           nVars = int(pkbin_vars[0][0])
           domainSize = int(pkbin_vars[1][0])

           self.scl_array_sh = RawArray(c_uint32, nVars * NWORDS_256BIT)     
           self.scl_array = np.frombuffer(
                     self.scl_array_sh, dtype=np.uint32).reshape((nVars, NWORDS_256BIT))
           if self.witness_f.endswith('.json'):
             f = open(self.witness_f,'r')
             np.copyto(
                 self.scl_array[:nVars],
                 np.reshape(
                     np.asarray(
                         [BigInt(c).as_uint256() for c in ast.literal_eval(json.dumps(json.load(f)))],
                                                                       dtype=np.uint32),(-1, NWORDS_256BIT))
                      )
             f.close()
           elif self.witness_f.endswith('.txt'):
             with open(self.witness_f, 'r') as f:
               np.copyto(
                 self.scl_array[:nVars],
                 np.reshape(
                     np.asarray(
                          [BigInt(c).as_uint256() for c in f]),(-1,NWORDS_256BIT))
                     )

           self.sorted_scl_array_idx_sh = RawArray(c_uint32, domainSize)
           self.sorted_scl_array_idx = np.frombuffer(self.sorted_scl_array_idx_sh, dtype=np.uint32)
          
           self.sorted_scl_array_sh = RawArray(c_uint32, (domainSize + 4) * NWORDS_256BIT)  # sorted witness + [one] + r/s or sorted polH
           self.sorted_scl_array = np.frombuffer(
                             self.sorted_scl_array_sh, dtype=np.uint32).reshape((domainSize+4, NWORDS_256BIT)
                                                )
       else:
          logging.error('Witness file %s doesn\'t exist', self.witness_f)
          os.exit(1)
       
    def load_pkdata(self):
       if self.proving_key_f.endswith('npz'):
          npzfile = np.load(self.proving_key_f)
          self.pk['protocol'] = np.uint32(PROTOCOL_T_GROTH)
          self.pk['Rbitlen']  = np.asarray(ZField.get_reduction_data()['Rbitlen'],dtype=np.uint32)
          self.pk['k_binformat'] = np.uint32(FMT_MONT)
          self.pk['k_ecformat'] = np.uint32(EC_T_AFFINE)
          self.pk['alfa_1'] = npzfile['alfa_1_u256']
          self.pk['beta_1'] = npzfile['beta_1_u256']
          self.pk['delta_1']= npzfile['delta_1_u256']
          self.pk['beta_2'] = npzfile['beta_2_u256']
          self.pk['delta_2'] = npzfile['delta_2_u256']

          self.pk['A'] = npzfile['A_u256']
          self.pk['B1'] = npzfile['B1_u256']
          self.pk['B2'] = npzfile['B2_u256']
          self.pk['C']  = npzfile['C_u256']
          self.pk['hExps'] = npzfile['hExps_u256']
          self.pk['polsA'] = npzfile['polsA_u256']
          self.pk['polsB'] = npzfile['polsB_u256']
          self.pk['polsC'] = npzfile['polsC_u256']
          self.pk['nVars'] = npzfile['nvars']
          self.pk['nPublic'] = npzfile['npublic']
          self.pk['domainSize'] = npzfile['domain_size']

       ## Open and parse proving key data
       elif self.proving_key_f.endswith('.json'):
           f = open(self.proving_key_f,'r')
           tmp_data = json.load(f)
           vk_proof = json_to_dict(tmp_data)
           f.close()
           self.pk = pkjson_to_vars(vk_proof, self.proving_key_f)  
           del vk_proof
           del tmp_data

       elif self.proving_key_f.endswith('.bin'):
          pk_bin = readU256PKFile_h(self.proving_key_f.encode("UTF-8"))
          self.pk = pkbin_to_vars(pk_bin)
          del pk_bin

       logging.info('')
       logging.info('')
       logging.info('#################################### ')
       logging.info(' - nVars      : %s', self.pk['nVars'])
       logging.info(' - nPublic    : %s', self.pk['nPublic'])
       logging.info(' - domainBits : %s', self.pk['domainBits'])
       logging.info('#################################### ')
       logging.info('')
       logging.info('')

    def timeStats(self, t):
      for s in t:
        for k, v in s.items():
           if k is not 'total':
             s[k] = str(round(v,2)) + '(' + str(round(100*v/s['total'],2)) + '%)'
   
    def logTimeResults(self):
     
      self.t_EC['total'] = round(self.t_EC['total.0'] + self.t_EC['total.1'] + self.t_EC['EC Mexp2'],2)
      self.timeStats([self.t_EC, self.t_GP, self.t_P])
      self.t_P['total'] = round(self.t_P['total'],2)
      self.t_GP['total'] = round(self.t_GP['total'],2)

      logging.info('')
      logging.info('')
      logging.info('#################################### ')
      logging.info('Total Time to generate proof : %s seconds', self.t_GP['total'])
      logging.info('')
      logging.info('------ Time EC [sec] : %s ', str(round(self.t_EC['total'],2)) + '(' + str(round(100*self.t_EC['total']/self.t_GP['total'],2)) + '%)')
      logging.info('%s', self.t_EC)
      logging.info('')
      logging.info('----- Time Poly [sec] : %s ', str(round(self.t_P['total'],2)) + '(' + str(round(100*self.t_P['total']/self.t_GP['total'],2)) + '%)')
      logging.info('%s', self.t_P)
      logging.info('')
      logging.info('----- Time GP [sec] : %s ', self.t_GP['total'])
      logging.info('%s', self.t_GP)
      logging.info('#################################### ')
      logging.info('')
      logging.info('')

    def proof(self, witness_f, mproc = False):
      self.witness_f = witness_f
      logging.info('#################################### ')
      logging.info("Starting proof...")

      self.gen_proof(mproc)
       
      logging.info("Proof completed" )
      logging.info('#################################### ')

      self.logTimeResults()

      # convert data to pkvars if necessary (only if test_f is set)
      if self.test_f is not None:
        self.pk =  pkbin_to_vars(self.pk_short)
        del self.pk_short

      self.write_pdata()
      self.write_proof()
      self.test_results()

      copy_input_files([self.out_proof_f, self.out_public_f, self.out_proving_key_f, witness_f],self.keep_f)

    def test_results(self):
      proof_r = True
      snarkjs = {}
      snarkjs['verify'] = 0
      if self.test_f is not None:
        logging.info("Calling snarkjs proof to compare computed proving and verification keys")
        # Write rand json
        randout_dict={}
        randout_dict['r'] = str(BigInt.from_uint256(self.r_scl).as_long())
        randout_dict['s'] = str(BigInt.from_uint256(self.s_scl).as_long())
        randout_json = json.dumps(randout_dict, indent=4, sort_keys=True)
        f = open(self.test_f, 'w')
        print(randout_json, file=f)
        f.close()
        snarkjs = self.launch_snarkjs("proof")
        p_r = pysnarks_compare(self.out_proof_f, snarkjs['p_f'], ['pi_a', 'pi_b', 'pi_c'],0)
        pd_r = pysnarks_compare(self.out_public_f, snarkjs['pd_f'], None, 0)
        proof_r = p_r and pd_r
        if proof_r:
          logging.info("Compared keys are equal")
        elif p_r:
          logging.info("Verification keys are different")
        elif pd_r:
          logging.info("Proving keys are different")
        else:
          logging.info("Verification and Proving keys are different")

      if self.verify_en:
        logging.info('#################################### ')
        logging.info("Calling snarkjs verify to verify proof ....")
        logging.info("")
        snarkjs = self.launch_snarkjs("verify")
        if snarkjs['verify'] == 0:
          logging.info("Verification SUCCEDED")
        else:
          logging.info("Verification FAILED")
        logging.info('#################################### ')

      return

    def launch_snarkjs(self, mode):
        snarkjs = {}
        if mode=="proof" :
          snarkjs['p_f'] = self.keep_f + '/' +  'tmp_p_f.json'
          snarkjs['pd_f'] = self.keep_f + '/' + 'tmp_pd_f.json'
          # snarkjs setup is launched with circuit.json, format extended. Convert input file if necessary
          if self.witness_f.endswith('.json') and self.proving_key_f.endswith('.json') and self.pk['k_binformat']==FMT_EXT :
             witness_file = self.witness_f
             proving_key_file = self.proving_key_f
          elif  self.witness_f.endswith('.json'):
            if self.proving_key_f.endswith('.bin'):
               witness_file = self.witness_f
               proving_key_file = self.proving_key_f.replace('.bin','_cpy.json')
            else: 
               witness_file = self.witness_f
               proving_key_file = self.proving_key_f.replace('.json','_cpy.json')
            proving_key_file = self.keep_f + '/' + proving_key_file.rsplit('/', 1)[1]
            #proving_key_file = self.keep_f + proving_key_file
            pk_dict = pkvars_to_json(FMT_EXT,EC_T_AFFINE, self.pk)
            pk_json = json.dumps(pk_dict, indent=4, sort_keys=True)
            f = open(proving_key_file, 'w')
            print(pk_json, file=f)
            f.close()
          else:
            logging.error(' Witness file %s needs to be .json', self.witness_f)
            sys.exit(1)

          if self.test_f is not None:
             debug_command = "--d"
             debug_file = self.test_f
          else:
             debug_command = ""
             debug_file = ""

          call([self.snarkjs, "proof", "-w", witness_file, "--pk", proving_key_file, "-p", snarkjs['p_f'],"--pub",snarkjs['pd_f'], debug_command, debug_file])

        elif mode == "verify":
          snarkjs['p_f'] = self.out_proof_f
          snarkjs['pd_f'] = self.out_public_f
          if self.verification_key_f is not None and self.verification_key_f.endswith('.json') :
             verification_key_file = self.verification_key_f
          else :
             logging.error(' To launch snarkjs, verification file %s needs to be a json file', self.verification_key_f)
             os.exit(1)
        
          snarkjs['verify'] = call([self.snarkjs, "verify", "--vk", verification_key_file, "-p", snarkjs['p_f'],"--pub",snarkjs['pd_f']])
       
        return snarkjs

    def gen_proof(self, mproc=False):
        """
          public_signals, pi_a_eccf1, pi_b_eccf2, pi_c_eccf1 

            inputs  : Witness from file 
            Outputs : Bin witnes (2)

          1 - Read all witness data -> CPU. When batch samples available,
                  start process 3

          ------------------------
            inputs  : Witness (1)
            Outputs : pi's (final result),  pi_c (4).

          2.1 - get witness batch  
          2.2 - sort witness1 batch [0-nVars] -> CPU
          2.3 - EC multi expo with batch A/B1/B2 and witnesss1 -> pi_a, pi_b, pib -> GPU
          2.4 - sort witness2 batch [nPublic+1,nVars]
          2.5 - EC multi expo with batch C and witnesss2 -> pi_c
          ------------------------
       
            inputs  : pols from file
                      witness (1). But reqiures all witness read
            Outputs : pX_T (I can't start 4 until all poool is availlable)

          3.1  Evaluate pols pA, pB and pC. Complete poly needs to be evaluated
                 before starting FFTs. CPU cores can work in this. Ideally,
                 multiple cores help evaluate pA. When it is done, repeat for pB,...
                 polsA   ----> polA_T  ----> delete polsA. I need mutexes. Modify 
                 C routine to do this
          ------------------------

            inputs  : pX_T (3)A
            output  : pH at the end of process (5). 

          4.1   Get polA_T, polB_T and polB_T batch
          4.2   perform 3xIFFT, 1 poly MUL and 1 poly SUB. Dedicate GPUs to do partial IFFT
                and CPU to combine results to do 3-step fft. Complete  IFFT-A, IFFT-B, IFFT-C,
                 FFT-A, FFT-B, FFT-AxFFT-B, IFFTAxB, SUB and finaly get pH
          -----------------------
            
            Input : pH (4), hExps (file), pi_c (2)
            Output : pi_c

          5.1 - EC multi expo with batch hExps and pH -
 
         
         ------
          
            1 1 1 1 1 1 1 1 1 
                  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
                            3 3 3 3 3 3 pA done 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                               StartIFFT PA 4 4 4 4 4 4 4 4 4 4 4 4 PH Done
                                                                                     5 5 5 5 5 5 5 5 5 5 5 5 5 5 



        """
        ######################
        # Beginning of P1 - Read Witness
        ######################
        start = time.time()
        start_p = time.time()

        # Read witness info
        self.read_witness_data()

        end = time.time()
        self.t_GP['read w'] = end - start

        
        ######################
        # Beginning of P2 
        #   - Get witness batch, sort and EC Multiexp
        ######################
        start = time.time()
        ZField.set_field(MOD_FIELD)
        # Init r and s scalars
        self.r_scl = BigInt(random.randint(1,ZField.get_extended_p().as_long()-1)).as_uint256()
        self.s_scl = BigInt(random.randint(1,ZField.get_extended_p().as_long()-1)).as_uint256()

        pk_bin = pkbin_get(self.pk,['nVars', 'nPublic', 'domainSize','hExps', 'delta_1','polsA'])

        #large input : hExps. I can overwrite it provided that i don't require comparing to snarkjs
        nVars = pk_bin[0][0]
        nPublic = pk_bin[1][0]
        domainSize = pk_bin[2][0]
        ds_1 = domainSize - 1
        hExps = np.reshape(pk_bin[3],(-1,NWORDS_256BIT))
        delta_1 = pk_bin[4]
        polH = np.reshape(pk_bin[5][:(domainSize-1)*NWORDS_256BIT],((domainSize-1),NWORDS_256BIT))

        
        # sorted scl array is in shared memory. It keeps a working window for data
        np.copyto(
            self.sorted_scl_array_idx[:nPublic+1],
            sortu256_idx_h(self.scl_array[:nPublic+1])
                 )

        np.copyto(
          self.sorted_scl_array[:nPublic+1],
          self.scl_array[:nPublic+1][self.sorted_scl_array_idx[:nPublic+1]]
                 )

        end = time.time()
        self.t_GP['sort1.0'] = end - start

        start = time.time()

        self.findECPoints(0)

        end = time.time()
        self.t_GP['EC Mexp1.0'] = end - start

        np.copyto(
            self.sorted_scl_array_idx[nPublic+1:nVars],
            sortu256_idx_h(self.scl_array[nPublic+1:nVars])
                 )

        np.copyto(
          self.sorted_scl_array[nPublic+1:nVars],
          self.scl_array[nPublic+1:nVars][self.sorted_scl_array_idx[nPublic+1:nVars]]
                 )

        self.sorted_scl_array[nPublic:nPublic+1] = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32).reshape((-1,NWORDS_256BIT))
        self.sorted_scl_array[nVars:nVars+1] = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32).reshape((-1,NWORDS_256BIT))
        self.sorted_scl_array[nVars+1:nVars+2] = self.r_scl

        end = time.time()
        self.t_GP['sort1.1'] = end - start

        start = time.time()

        self.findECPoints(1)

        end = time.time()
        self.t_GP['EC Mexp1.1'] = end - start
        ######################
        # Beginning of P3 and P4
        #  P3 - Poly Eval
        #  P4 - Poly Operations
        ######################
        start = time.time()

        self.calculateH()

        end = time.time()
        self.t_GP['pH'] = end - start

        ######################
        # Beginning of P5
        #   - Final EC MultiExp
        ######################
        start = time.time()
        ZField.set_field(MOD_FIELD)
        r_mont = to_montgomeryN_h(np.reshape(self.r_scl,-1),MOD_FIELD)
        self.sorted_scl_array[ds_1+3:ds_1+4] = montmult_neg_h(
                                           np.reshape(r_mont,-1),
                                           np.reshape(self.s_scl,-1), MOD_FIELD
                                                             )

        ZField.set_field(MOD_GROUP)

        np.copyto(
               self.sorted_scl_array_idx[:ds_1],
               sortu256_idx_h(polH[:ds_1])
                 )

        np.copyto(
               self.sorted_scl_array[:ds_1],
               polH[:ds_1][self.sorted_scl_array_idx[:ds_1]]
                 )
        self.sorted_scl_array[ds_1:ds_1+1] = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32).reshape((-1,NWORDS_256BIT))
        self.sorted_scl_array[ds_1+1:ds_1+2] = self.s_scl
        self.sorted_scl_array[ds_1+2:ds_1+3] = self.r_scl

        np.copyto(
            hExps[:2*ds_1],
            np.reshape(
                np.reshape(hExps[:2*ds_1], 
                  (-1,2,NWORDS_256BIT))[self.sorted_scl_array_idx[:ds_1]],(-1,NWORDS_256BIT))
                     )

        hExps[2*ds_1:2*ds_1+2] = self.pi_c_eccf1[:2]
        hExps[2*ds_1+2:2*ds_1+4] = self.pi_a_eccf1[:2]
        hExps[2*ds_1+4:2*ds_1+6] = self.pib1_eccf1[:2]
        hExps[2*ds_1+6:2*ds_1+8] = np.reshape(delta_1,(-1,NWORDS_256BIT))

        self.pi_c_eccf1,t1 = self.compute_proof_ecp(
            self.ecbn128,
            self.sorted_scl_array[:ds_1+4],
            hExps[:2*ds_1+8],
            False)

        end = time.time()
        self.t_EC['EC Mexp2'] = end - start
        self.t_GP['EC Mexp2'] = end - start

        self.public_signals = np.copy(self.scl_array[1:nPublic+1])

        self.t_GP['total'] = end - start_p

        return 
 

    def compute_proof_ecp(self, pyCuOjb, K, P, ec2):
            ZField.set_field(MOD_GROUP)
            ecbn128_samples = np.concatenate((K,np.reshape(P,(-1,NWORDS_256BIT))))
            ecp,t = ec_mad_cuda(pyCuOjb, ecbn128_samples, MOD_GROUP, ec2)
            if ec2:
              ecp = ec2_jac2aff_h(ecp.reshape(-1),MOD_GROUP)
            else:
              ecp = ec_jac2aff_h(ecp.reshape(-1),MOD_GROUP)

            return ecp, t

    def findECPoints(self, phase):
    
        start_ec = time.time()

        ZField.set_field(MOD_GROUP)

        pk_bin = pkbin_get(self.pk,['nVars', 'nPublic','A', 'B1', 'B2', 'C'])
        nVars = pk_bin[0][0]
        nPublic = pk_bin[1][0]
        A = pk_bin[2]
        B1 = pk_bin[3]
        B2 = pk_bin[4]
        C  = pk_bin[5]

        if phase == 0:
          start_idx = 0
          start_idx2 = start_idx
          end_idx = nPublic+1
          end_idx2 = end_idx
        
        else:
          start_idx = nPublic+1
          start_idx2 = nPublic
          end_idx = nVars
          end_idx2 = nVars+2
       

        #pi_a -> add 1 and r_u256 to scl, and alpha1 and delta1 to P
        
        np.copyto(
            A[2*start_idx*NWORDS_256BIT:2*end_idx*NWORDS_256BIT],
            np.reshape(
                 np.reshape(A[2*start_idx*NWORDS_256BIT:2*end_idx*NWORDS_256BIT],
                           (-1,2,NWORDS_256BIT))[self.sorted_scl_array_idx[start_idx:end_idx]],-1)
                 )

        self.pi_a_eccf1,t1 = self.compute_proof_ecp(
                                        self.ecbn128,
                                        self.sorted_scl_array[start_idx2:end_idx2],
                                        A[2*start_idx2*NWORDS_256BIT:2*end_idx2*NWORDS_256BIT],
                                        False)
        # Copy result for carrying sum
        np.copyto(
             A[2*(end_idx-1)*NWORDS_256BIT:2*end_idx*NWORDS_256BIT], 
             np.reshape(self.pi_a_eccf1[:2],-1)
         )
        self.t_EC['pi_a.'+str(phase)] = t1

        self.sorted_scl_array[nVars+1:nVars+2] = self.s_scl

        np.copyto(
            B2[4*start_idx*NWORDS_256BIT:4*end_idx*NWORDS_256BIT],
            np.reshape(
                  np.reshape(B2[4*start_idx*NWORDS_256BIT:4*end_idx*NWORDS_256BIT], 
                               (-1,4,NWORDS_256BIT))[self.sorted_scl_array_idx[start_idx:end_idx]],-1)
           )

        self.pi_b_eccf2,t1 = self.compute_proof_ecp(
                                         self.ec2bn128, 
                                         self.sorted_scl_array[start_idx2:end_idx2],
                                         B2[4*start_idx2*NWORDS_256BIT:4*end_idx2*NWORDS_256BIT],
                                         True)
        # Copy result for carrying sum
        np.copyto(
             B2[4*(end_idx-1)*NWORDS_256BIT:4*end_idx*NWORDS_256BIT], 
             np.reshape(self.pi_b_eccf2[:4],-1)
         )
        self.t_EC['pi_b.'+str(phase)]= t1

        np.copyto(
             B1[2*start_idx*NWORDS_256BIT:2*end_idx*NWORDS_256BIT],
             np.reshape(
                  np.reshape(B1[2*start_idx*NWORDS_256BIT:2*end_idx*NWORDS_256BIT],
                                          (-1,2,NWORDS_256BIT))[self.sorted_scl_array_idx[start_idx:end_idx]],-1)
            )

        self.pib1_eccf1,t1 = self.compute_proof_ecp(
                                          self.ecbn128,
                                          self.sorted_scl_array[start_idx2:end_idx2], 
                                          B1[2*start_idx2*NWORDS_256BIT:2*end_idx2*NWORDS_256BIT],
                                          False)
        # Copy result for carrying sum
        np.copyto(
             B1[2*(end_idx-1)*NWORDS_256BIT:2*end_idx*NWORDS_256BIT], 
             np.reshape(self.pib1_eccf1[:2],-1)
         )
        self.t_EC['pib1.'+str(phase)] = t1

        if phase == 1:
           np.copyto(
               C[2*start_idx*NWORDS_256BIT:2*end_idx*NWORDS_256BIT], 
               np.reshape(
                  np.reshape(C[2*start_idx*NWORDS_256BIT:2*end_idx*NWORDS_256BIT],
                                           (-1,2,NWORDS_256BIT))[self.sorted_scl_array_idx[start_idx:end_idx]],-1)
             )

           self.pi_c_eccf1,t1 = self.compute_proof_ecp(
                                               self.ecbn128,
                                               self.sorted_scl_array[start_idx:end_idx],
                                               C[2*start_idx*NWORDS_256BIT:2*end_idx*NWORDS_256BIT],
                                               False)
           self.t_EC['pi_c.'+str(phase)] = t1

        end_ec = time.time()
        self.t_EC['total.'+str(phase)]  = end_ec - start_ec

        return 


    def write_pdata(self):
        if self.out_public_f.endswith('.json'):
           # Write public file
           ZField.set_field(MOD_FIELD)
           ps = [str(BigInt.from_uint256(el).as_long()) for el in self.public_signals]
           j = json.dumps(ps, indent=4)
           f = open(self.out_public_f, 'w')
           print(j,file=f)
           f.close()
        elif self.out_public_f.endswith('bin') :
           public_bin = np.concatenate((
                   np.asarray([self.public_signals.shape[0]], dtype=np.uint32),
                   np.reshape(self.public_signals,(-1))))
           writeU256DataFile_h(public_bin, self.out_public_f.encode("UTF-8"))

    def write_proof(self):
        if self.out_proof_f.endswith('.json'):
           ZField.set_field(MOD_GROUP)
           # write proof file
           P = ECC.from_uint256(self.pi_a_eccf1, in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)
           pi_a = [el.extend().as_str() for el in P][0]
   
           P = ECC.from_uint256(self.pi_c_eccf1, in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)
           pi_c = [el.extend().as_str() for el in P][0]
   
           P = ECC.from_uint256(np.reshape(self.pi_b_eccf2,(-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True, ec2=True)
           pi_b = [el.extend().as_str() for el in P][0]
           proof = {"pi_a" : pi_a, "pi_b" : pi_b, "pi_c" : pi_c, "protocol" : "groth"}
           j = json.dumps(proof, indent=4, sort_keys=True)
           f = open(self.out_proof_f, 'w')
           print(j, file=f)
           f.close()
        elif self.out_proof_f.endswith('.bin'):
           self.pi_a_eccf1 = from_montgomeryN_h(np.reshape(self.pi_a_eccf1,(-1)), MOD_GROUP)
           self.pi_b_eccf2 = from_montgomeryN_h(np.reshape(self.pi_b_eccf2,(-1)), MOD_GROUP)
           self.pi_c_eccf1 = from_montgomeryN_h(np.reshape(self.pi_c_eccf1,(-1)), MOD_GROUP)
           proof_bin = np.concatenate((
                    self.pi_a_eccf1, 
                    self.pi_b_eccf2,
                    self.pi_c_eccf1))
           writeU256DataFile_h(proof_bin, self.out_public_f.encode("UTF-8"))
               

    def evalPoly(self, pX, nVars, m):
        # Convert witness to montgomery in zpoly_maddm_h
        #polA_T, polB_T, polC_T are montgomery -> polsA_sps_u256, polsB_sps_u256, polsC_sps_u256 are montgomery
        pidx = ZField.get_field()
        reduce_coeff = 0  
        polX_T = mpoly_eval_h(self.scl_array[:nVars],np.reshape(pX,-1), reduce_coeff, m, 0, nVars, mp.cpu_count(), pidx)
        np.copyto(pX, polX_T)


    def calculateH(self):

        start_h = time.time()

        ZField.set_field(MOD_FIELD)
        pk_bin = pkbin_get(self.pk,['nVars', 'domainSize', 'polsA', 'polsB', 'polsC', 'polsH'])
        nVars = pk_bin[0][0]
        m = pk_bin[1][0]
        pA = np.reshape(pk_bin[2][:m*NWORDS_256BIT],(m,NWORDS_256BIT))
        pB = np.reshape(pk_bin[3][:m*NWORDS_256BIT],(m,NWORDS_256BIT))
        pC = np.reshape(pk_bin[4][:m*NWORDS_256BIT],(m,NWORDS_256BIT))
        pH = np.reshape(pk_bin[5][:(2*m-1)*NWORDS_256BIT],((2*m-1),NWORDS_256BIT))

        # Convert witness to montgomery in zpoly_maddm_h
        #polA_T, polB_T, polC_T are montgomery -> polsA_sps_u256, polsB_sps_u256, polsC_sps_u256 are montgomery
        start = time.time()

        self.evalPoly(pA, nVars, m)
        self.evalPoly(pB, nVars, m)
        self.evalPoly(pC, nVars, m)

        end = time.time()
        self.t_P['eval'] = end-start

        ifft_params = ntt_build_h(pA.shape[0])

        if self.n_bits_roots < ifft_params['levels']:
          logging.error('Insufficient number of roots in ' + self.roots_f + 'Required number of roots is '+ str(1<< ifft_params['levels']))
          sys.exit(1)
     

        # polC_S  is extended -> use extended scaler
        polC_S,t1 = zpoly_ifft_cuda(self.cuzpoly, pC, ifft_params, ZField.get_field(), as_mont=0, roots=self.roots_rdc_u256)
        np.copyto(pC,polC_S)
        del polC_S
        self.t_P['ifft-C'] = t1

        # polA_S montgomery -> use montgomery scaler
        polA_S,t1 = zpoly_ifft_cuda(self.cuzpoly, pA,ifft_params, ZField.get_field(), as_mont=1)
        np.copyto(pA,polA_S)
        del polA_S
        self.t_P['ifft-A'] =  t1

        # polB_S montgomery  -> use montgomery scaler
        # TODO : return_val = 0, out_extra_len= out_len
        polB_S,t1 = zpoly_ifft_cuda(self.cuzpoly, pB, ifft_params, ZField.get_field(), as_mont=1, return_val = 1, out_extra_len=0)
        np.copyto(pB,polB_S)
        del polB_S
        self.t_P['ifft-B'] = t1

        mul_params = ntt_build_h(pH.shape[0])
        #polAB_S is extended -> use extended scaler
        # TODO : polB_S is stored in device mem already from previous operation. Do not return  value
        polAB_S,t1 = zpoly_mul_cuda(self.cuzpoly, pA,pB,mul_params, ZField.get_field(), roots=self.roots_rdc_u256, return_val=1, as_mont=0)
        nsamplesH = zpoly_norm_h(polAB_S)
        np.copyto(pH[:nsamplesH],polAB_S[:nsamplesH])
        del polAB_S
        self.t_P['mul'] = t1

        # polABC_S is extended
        # TODO : polAB_S is stored in device moem already from previous operatoin. Do not return value.
        # TODO : perform several sub operations per thread to improve efficiency
        polABC_S,t1 = zpoly_sub_cuda(self.cuzpoly, pH[:nsamplesH], pC, ZField.get_field(), vectorA_len = 0, return_val=1)
        np.copyto(pH[:m-1],polABC_S[m:])
        del polABC_S
        self.t_P['sub'] =  t1

        end_h = time.time()
        self.t_P['total'] = end_h - start_h

        return
