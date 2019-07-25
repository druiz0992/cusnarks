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
from cuda_wrapper import *

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

ROOTS_1M_filename = '../../data/zpoly_roots_1M.bin'

class GrothProver(object):
    
    def __init__(self, proving_key_f, verification_key_f=None,curve='BN128',  out_proof_f = None,
                 out_public_f = None, out_pk_f=None, out_pk_format=FMT_MONT,
                 out_proof_format= FMT_EXT, out_public_format=FMT_EXT, test_f=None, 
                 benchmark_f=None, seed=None, snarkjs=None, accel = True, verify_en=0, keep_f=None):

        self.accel = accel

        if seed is not None:
          self.seed = seed
          random.seed(seed) 
        else:
          self.seed = random.randint(0,1<<32)

        if use_pycusnarks and self.accel:
          self.roots1M_rdc_u256_sh = RawArray(c_uint32,  (1 << 20) * NWORDS_256BIT)
          self.roots1M_rdc_u256 = np.frombuffer(self.roots1M_rdc_u256_sh, dtype=np.uint32).reshape((1<<20, NWORDS_256BIT))
          np.copyto(self.roots1M_rdc_u256, readU256DataFile_h(ROOTS_1M_filename.encode("UTF-8"), 1<<20, 1<<20) )
          #self.roots1M_rdc_u256 = readU256DataFile_h(ROOTS_1M_filename.encode("UTF-8"), 1<<20, 1<<20)

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
        self.public_signals = None
        self.witness_f = None
        self.snarkjs = snarkjs
        self.verify_en = verify_en
        
        #scalar array : extended witness / polH
        self.scl_array = None
        self.sorted_scl_array = None
        self.sorted_scl_array_idx = None

        if keep_f is None:
            print ("Repo directory needs to be provided\n")
            sys.exit(1)
        self.keep_f = gen_reponame(keep_f, sufix="_PROVER")
        copy_input_files([proving_key_f, verification_key_f], self.keep_f)
        if test_f is None:
           self.test_f = test_f
        else:
           self.test_f= self.keep_f + '/' + test_f

        logging.basicConfig(filename=self.keep_f + '/log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')
        logging.info('#################################### ')
        logging.info('Staring Groth prover with the follwing parameters :')
        logging.info(' - proving_key_f : %s', proving_key_f)
        logging.info(' - verification_key_f : %s',verification_key_f)
        logging.info(' - curve : %s',curve)
        logging.info(' - out_proof_f : %s',out_proof_f)
        logging.info(' - out_pk_f : %s',out_pk_f)
        logging.info(' - out_pk_format : %s',out_pk_format) 
        logging.info(' - out_proof_format : %s',out_proof_format)
        logging.info(' - out_public_format :  %s',out_public_format)
        logging.info(' - test_f : %s',self.test_f)
        logging.info(' - benchmark_f : %s', benchmark_f)
        logging.info(' - seed : %s', seed)
        logging.info(' - snarkjs : %s', snarkjs)
        logging.info(' - accel :  %s',accel)
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
               del pk_dic
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
        if use_pycusnarks and self.accel:
             #self.pk = pkvars_to_bin(FMT_MONT, EC_T_AFFINE, self.pk, ext=True)
             if self.test_f :
               self.pk_short = pkvars_to_bin(FMT_MONT, EC_T_AFFINE, self.pk, ext=False)

             pk_bin = pkvars_to_bin(FMT_MONT, EC_T_AFFINE, self.pk, ext=True)
             self.pk_sh = RawArray(c_uint32, int(pk_bin[0]))
             self.pk = np.frombuffer(self.pk_sh, dtype=np.uint32)
             np.copyto(self.pk, pk_bin)
             del pk_bin
             


    def read_witness_data(self):
       ## Open and parse witness data
       if os.path.isfile(self.witness_f):
           f = open(self.witness_f,'r')
           if use_pycusnarks and self.accel:
              pkbin_vars = pkbin_get(self.pk,['nVars','domainSize'])
              nVars = int(pkbin_vars[0][0])
              domainSize = int(pkbin_vars[1][0])
              self.scl_array_sh = RawArray(c_uint32, nVars * NWORDS_256BIT)     
              self.scl_array = np.frombuffer(self.scl_array_sh, dtype=np.uint32).reshape((nVars, NWORDS_256BIT))
              np.copyto(self.scl_array[:nVars], np.reshape(np.asarray([BigInt(c).as_uint256() for c in ast.literal_eval(json.dumps(json.load(f)))],
                                                        dtype=np.uint32),(-1, NWORDS_256BIT)) )

              self.sorted_scl_array_idx_sh = RawArray(c_uint32, domainSize)
              self.sorted_scl_array_idx = np.frombuffer(self.sorted_scl_array_idx_sh, dtype=np.uint32)
          
              self.sorted_scl_array_sh = RawArray(c_uint32, (domainSize + 4) * NWORDS_256BIT)  # sorted witness + [one] + r/s or sorted polH
              self.sorted_scl_array = np.frombuffer(self.sorted_scl_array_sh, dtype=np.uint32).reshape((domainSize+4, NWORDS_256BIT))
              #self.scl_array = np.reshape(np.asarray([BigInt(c).as_uint256() for c in ast.literal_eval(json.dumps(json.load(f)))],
              #                                          dtype=np.uint32),(-1, NWORDS_256BIT))
           else:
              self.scl_array = [BigInt(c) for c in ast.literal_eval(json.dumps(json.load(f)))]
           f.close()
       else:
          logging.error('Witness file %s doesn\'t exist', self.witness_f)
          os.exit(1)
       
    def load_pkdata(self):
       if use_pycusnarks and self.accel and self.proving_key_f.endswith('npz'):
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
           if use_pycusnarks and self.accel:
              self.pk = pkjson_to_vars(vk_proof, self.proving_key_f)  
           else:
              ZField.find_roots(ZUtils.NROOTS)
              self.pk = pkjson_to_pyvars(vk_proof)  
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

    def proof(self, witness_f):
      self.witness_f = witness_f
      logging.info("Starting proof...")

      t1, t2, t3 = self.gen_proof()
       
      logging.info("Proof completed" )
      logging.info('')
      logging.info('')
      logging.info('#################################### ')
      logging.info('')
      logging.info('Total Time to generate proof : %s seconds', t3['total'])
      logging.info('')
      logging.info('------ Time EC [sec] : %s ', t1['total'] + t1['pi_c final'])
      logging.info('%s', t1)
      logging.info('')
      logging.info('----- Time FFT [sec] : %s ', t2['total'])
      logging.info('%s', t2)
      logging.info('')
      logging.info('----- Time Main [sec] : %s ', t3['total'])
      logging.info('%s', t3)
      logging.info('')
      logging.info('#################################### ')
      logging.info('')
      logging.info('')

      # convert data to pkvars if necessary (only if test_f is set)
      if use_pycusnarks and self.accel and self.test_f is not None:
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
        logging.info("Calling snarkjs verify to verify proof ....")
        logging.info("")
        snarkjs = self.launch_snarkjs("verify")
        if snarkjs['verify'] == 0:
          logging.info("Verification SUCCEDED")
        else:
          logging.info("Verification FAILED")
        logging.info("")
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

    def gen_proof(self):
        """
          public_signals, pi_a_eccf1, pi_b_eccf2, pi_c_eccf1 
        """
        # Read witness
        self.t_GP = {}
        start = time.time()
        start_p = time.time()
        self.read_witness_data()
        end = time.time()
        self.t_GP['read w'] = round(end - start,2)

        ZField.set_field(MOD_FIELD)
        # Init r and s scalars
        self.r_scl = BigInt(random.randint(1,ZField.get_extended_p().as_long()-1))
        self.s_scl = BigInt(random.randint(1,ZField.get_extended_p().as_long()-1))

        if use_pycusnarks and self.accel:
           self.r_scl = self.r_scl.as_uint256()
           self.s_scl = self.s_scl.as_uint256()
           pk_bin = pkbin_get(self.pk,['nVars', 'nPublic', 'domainSize','hExps', 'delta_1','polsA'])
           #large input : hExps. I can overwrite it provided that i don't require comparing to snarkjs
           nVars = pk_bin[0][0]
           nPublic = pk_bin[1][0]
           domainSize = pk_bin[2][0]
           hExps = np.reshape(pk_bin[3],(-1,NWORDS_256BIT))
           delta_1 = pk_bin[4]
           polH = np.reshape(pk_bin[5][:(domainSize-1)*NWORDS_256BIT],((domainSize-1),NWORDS_256BIT))

           np.copyto(self.sorted_scl_array_idx[:nVars], sortu256_idx_h(self.scl_array[:nVars]))
           np.copyto(self.sorted_scl_array[:nVars], self.scl_array[:nVars][self.sorted_scl_array_idx[:nVars]])

        end = time.time()
        self.t_GP['sort'] = round(end - start,2)


        # Accumulate multiplication of S EC points and S scalar. Parallelization
        # can be accomplised by each thread performing a multiplication and storing
        # values in same input EC point
        # Second step is to add all ECC points
        pib1_eccf1 = self.findECPoints()

        d1 = d2 = d3 = 0
        # polH must be in extended format
        
        if use_pycusnarks and self.accel:
          self.calculateH(d1, d2, d3)

          ds_1 = domainSize - 1
          start = time.time()
          ZField.set_field(MOD_FIELD)
          r_mont = to_montgomeryN_h(np.reshape(self.r_scl,-1),MOD_FIELD)
          self.sorted_scl_array[ds_1+3:ds_1+4] = montmult_neg_h(np.reshape(r_mont,-1),np.reshape(self.s_scl,-1), MOD_FIELD)

          ZField.set_field(MOD_GROUP)

          np.copyto(self.sorted_scl_array_idx[:ds_1], sortu256_idx_h(polH[:ds_1]))
          np.copyto(self.sorted_scl_array[:ds_1], polH[:ds_1][self.sorted_scl_array_idx[:ds_1]])
          self.sorted_scl_array[ds_1:ds_1+1] = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32).reshape((-1,NWORDS_256BIT))
          self.sorted_scl_array[ds_1+1:ds_1+2] = self.s_scl
          self.sorted_scl_array[ds_1+2:ds_1+3] = self.r_scl

          np.copyto(hExps[:2*ds_1], np.reshape(np.reshape(hExps[:2*ds_1], (-1,2,NWORDS_256BIT))[self.sorted_scl_array_idx[:ds_1]],(-1,NWORDS_256BIT)))
          hExps[2*ds_1:2*ds_1+2] = self.pi_c_eccf1[:2]
          hExps[2*ds_1+2:2*ds_1+4] = self.pi_a_eccf1[:2]
          hExps[2*ds_1+4:2*ds_1+6] = pib1_eccf1[:2]
          hExps[2*ds_1+6:2*ds_1+8] = np.reshape(delta_1,(-1,NWORDS_256BIT))
          self.pi_c_eccf1,t1 = self.compute_proof_ecp(self.ecbn128,
              self.sorted_scl_array[:ds_1+4],
              hExps[:2*ds_1+8],
              False)

          self.t_EC['pi_c final'] = round(t1,2)

          self.public_signals = np.copy(self.scl_array[1:nPublic+1])
          end = time.time()
          self.t_GP['total'] = round(end - start_p,2)

        else :
          polH = self.calculateH(d1, d2, d3)
          d4  = ZFieldElExt(-(self.r_scl * self.s_scl))
          ZField.set_field(MOD_GROUP)
          d5  = d4 * self.pk['delta_1']

          coeffH = polH.get_coeff()

          # Accumulate products of S ecc points and S scalars (same as at the beginning)
          n_coeff_h = len(coeffH)
          self.pi_c_eccf1 += np.sum(np.multiply(self.pk['hExps'][:n_coeff_h ], coeffH[:n_coeff_h]))

        
          self.pi_c_eccf1  += (self.pi_a_eccf1 * self.s_scl) + (pib1_eccf1 * self.r_scl) + d5

          self.public_signals = self.scl_array[1:self.pk['nPublic']+1]

        return self.t_EC, self.t_P, self.t_GP
 

    def compute_proof_ecp(self, pyCuOjb, K, P, ec2):
            ZField.set_field(MOD_GROUP)
            ecbn128_samples = np.concatenate((K,np.reshape(P,(-1,NWORDS_256BIT))))
            ecp,t = ec_mad_cuda(pyCuOjb, ecbn128_samples, MOD_GROUP, ec2)
            if ec2:
              ecp = ec2_jac2aff_h(ecp.reshape(-1),MOD_GROUP)
            else:
              ecp = ec_jac2aff_h(ecp.reshape(-1),MOD_GROUP)

            return ecp, t

    def findECPoints(self):
    
        self.t_EC = {}
        ZField.set_field(MOD_GROUP)

        if use_pycusnarks and self.accel:
          pk_bin = pkbin_get(self.pk,['nVars', 'nPublic','A', 'B1', 'B2', 'C'])
          nVars = pk_bin[0][0]
          nPublic = pk_bin[1][0]
          A = pk_bin[2]
          B1 = pk_bin[3]
          B2 = pk_bin[4]
          C  = pk_bin[5]
          start_ec = time.time()

          #pi_a -> add 1 and r_u256 to scl, and alpha1 and delta1 to P
          
          self.sorted_scl_array[nVars:nVars+1] = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32).reshape((-1,NWORDS_256BIT))
          self.sorted_scl_array[nVars+1:nVars+2] = self.r_scl

          np.copyto(A[:2*nVars*NWORDS_256BIT], np.reshape(np.reshape(A[:2*nVars*NWORDS_256BIT], (-1,2,NWORDS_256BIT))[self.sorted_scl_array_idx[:nVars]],-1))
          self.pi_a_eccf1,t1 = self.compute_proof_ecp(self.ecbn128,
              self.sorted_scl_array[:2+nVars],
              A,
              False)
          self.t_EC['pi_a'] = round(t1,2)

          self.sorted_scl_array[nVars+1:nVars+2] = self.s_scl
          np.copyto(B2[:4*nVars*NWORDS_256BIT], np.reshape(np.reshape(B2[:4*nVars*NWORDS_256BIT], (-1,4,NWORDS_256BIT))[self.sorted_scl_array_idx[:nVars]],-1))
          self.pi_b_eccf2,t1 = self.compute_proof_ecp(self.ec2bn128, 
               self.sorted_scl_array[:2+nVars],
               B2,
               True)
          self.t_EC['pi_b']= round(t1,2)

          np.copyto(B1[:2*nVars*NWORDS_256BIT], np.reshape(np.reshape(B1[:2*nVars*NWORDS_256BIT], (-1,2,NWORDS_256BIT))[self.sorted_scl_array_idx[:nVars]],-1))
          pib1_eccf1,t1 = self.compute_proof_ecp(self.ecbn128,
              self.sorted_scl_array[:2+nVars], 
              B1,
              False)
          self.t_EC['pib1'] = round(t1,2)

          np.copyto(self.sorted_scl_array_idx[nPublic+1:nVars], sortu256_idx_h(self.scl_array[nPublic+1:nVars]))
          np.copyto(self.sorted_scl_array[nPublic+1:nVars], self.scl_array[nPublic+1:nVars][self.sorted_scl_array_idx[nPublic+1:nVars]])

          np.copyto(C[2*(nPublic+1)*NWORDS_256BIT:2*nVars*NWORDS_256BIT], 
                 np.reshape(np.reshape(C[2*(nPublic+1)*NWORDS_256BIT:2*nVars*NWORDS_256BIT],
                                (-1,2,NWORDS_256BIT))[self.sorted_scl_array_idx[nPublic+1:nVars]],-1))
          self.pi_c_eccf1,t1 = self.compute_proof_ecp(self.ecbn128,
              self.sorted_scl_array[nPublic+1:nVars],
              C[2*(nPublic+1)*NWORDS_256BIT:2*nVars*NWORDS_256BIT],
              False)
          self.t_EC['pi_c'] = round(t1,2)

          end_ec = time.time()
          self.t_EC['total']  = round(end_ec - start_ec,2)

        else :
          nVars = self.pk['nVars']
          nPublic = self.pk['nPublic']
          self.pi_a_eccf1  = np.sum(np.multiply(self.pk['A'][:nVars], self.scl_array[:nVars]))
          self.pi_b_eccf2  = np.sum(np.multiply(self.pk['B2'][:nVars], self.scl_array[:nVars]))
          pib1_eccf1       = np.sum(np.multiply(self.pk['B1'][:nVars], self.scl_array[:nVars]))
          self.pi_c_eccf1  = np.sum(np.multiply(self.pk['C'][nPublic+1:nVars], self.scl_array[nPublic+1:nVars]))

          # pi_a = pi_a + alfa1 + delta1 * r
          self.pi_a_eccf1  += self.pk['alfa_1']
          self.pi_a_eccf1  += (self.pk['delta_1'] * self.r_scl)

          # pi_b = pi_b + beta2 + delta2 * s
          self.pi_b_eccf2  += self.pk['beta_2']
          self.pi_b_eccf2  += (self.pk['delta_2'] * self.s_scl)
  
          # pib1 = pib1 + beta1 + delta1 * s
          pib1_eccf1 += self.pk['beta_1']
          pib1_eccf1 += (self.pk['delta_1'] * self.s_scl)

        return pib1_eccf1


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
               

    def calculateH(self, d1, d2, d3):
        ZField.set_field(MOD_FIELD)
        self.t_P = {}

        if use_pycusnarks and self.accel:
          pk_bin = pkbin_get(self.pk,['nVars', 'domainSize', 'polsA', 'polsB', 'polsC', 'polsH'])
          nVars = pk_bin[0][0]
          m = pk_bin[1][0]
          pA = np.reshape(pk_bin[2][:m*NWORDS_256BIT],(m,NWORDS_256BIT))
          pB = np.reshape(pk_bin[3][:m*NWORDS_256BIT],(m,NWORDS_256BIT))
          pC = np.reshape(pk_bin[4][:m*NWORDS_256BIT],(m,NWORDS_256BIT))
          pH = np.reshape(pk_bin[5][:(2*m-1)*NWORDS_256BIT],((2*m-1),NWORDS_256BIT))
          start_h = time.time()
          #start = time.time()

          start = time.time()
          pidx = ZField.get_field()
          # Convert witness to montgomery in zpoly_maddm_h
          #polA_T, polB_T, polC_T are montgomery -> polsA_sps_u256, polsB_sps_u256, polsC_sps_u256 are montgomery
          reduce_coeff = 0
          polA_T = mpoly_eval_h(self.scl_array[:nVars],np.reshape(pA,-1), reduce_coeff, m, nVars, pidx)
          np.copyto(pA, polA_T)
          del polA_T
          polB_T = mpoly_eval_h(self.scl_array[:nVars],np.reshape(pB,-1), reduce_coeff, m, nVars, pidx)
          np.copyto(pB, polB_T)
          del polB_T
          polC_T = mpoly_eval_h(self.scl_array[:nVars],np.reshape(pC,-1), reduce_coeff, m, nVars, pidx)
          np.copyto(pC, polC_T)
          del polC_T
          end = time.time()
          self.t_P['eval'] = round(end-start,2)
          end_h = time.time()
          t_h = end_h - start_h


          start_h = time.time()
          ifft_params = ntt_build_h(pA.shape[0])

          # polC_S  is extended -> use extended scaler
          polC_S,t1 = zpoly_ifft_cuda(self.cuzpoly, pC, ifft_params, ZField.get_field(), as_mont=0, roots=self.roots1M_rdc_u256)
          np.copyto(pC,polC_S)
          del polC_S
          self.t_P['ifft-C'] = round(t1,2)

          # polA_S montgomery -> use montgomery scaler
          polA_S,t1 = zpoly_ifft_cuda(self.cuzpoly, pA,ifft_params, ZField.get_field(), as_mont=1)
          np.copyto(pA,polA_S)
          del polA_S
          self.t_P['ifft-A'] =  round(t1,2)

          # polB_S montgomery  -> use montgomery scaler
          # TODO : return_val = 0, out_extra_len= out_len
          polB_S,t1 = zpoly_ifft_cuda(self.cuzpoly, pB, ifft_params, ZField.get_field(), as_mont=1, return_val = 1, out_extra_len=0)
          np.copyto(pB,polB_S)
          del polB_S
          self.t_P['ifft-B'] = round(t1,2)

          mul_params = ntt_build_h(pH.shape[0])
          #polAB_S is extended -> use extended scaler
          # TODO : polB_S is stored in device mem already from previous operation. Do not return  value
          polAB_S,t1 = zpoly_mul_cuda(self.cuzpoly, pA,pB,mul_params, ZField.get_field(), roots=self.roots1M_rdc_u256, return_val=1, as_mont=0)
          nsamplesH = zpoly_norm_h(polAB_S)
          np.copyto(pH[:nsamplesH],polAB_S[:nsamplesH])
          del polAB_S
          self.t_P['mul'] = round(t1,2)

          # polABC_S is extended
          # TODO : polAB_S is stored in device moem already from previous operatoin. Do not return value.
          # TODO : perform several sub operations per thread to improve efficiency
          polABC_S,t1 = zpoly_sub_cuda(self.cuzpoly, pH[:nsamplesH], pC, ZField.get_field(), vectorA_len = 0, return_val=1)
          np.copyto(pH[:m-1],polABC_S[m:])
          del polABC_S
          self.t_P['sub'] =  round(t1,2)

          # polABC_S, polH_S are extended
          #polH_S = polABC_S[m:]

          #TODO : d1, d2 and d3 assumed to be zero
          end_h = time.time()
          self.t_P['total'] = round( end_h - start_h + t_h,2)
  
        else:
          nVars = self.pk['nVars']
          m = np.int32(self.pk['domainSize'])
          # Init dense poly of degree m-1 (all zero)
          polA_T = ZPoly([ZFieldElExt(0) for i in xrange(m)])
          polB_T = ZPoly([ZFieldElExt(0) for i in xrange(m)])
          polC_T = ZPoly([ZFieldElExt(0) for i in xrange(m)])

          # interpretation of polsA is that there are up to S sparse polynomials, and each sparse poly
          # has C sparse coeffs
          # polA_T = polA_T + witness[s] * polsA[s]
          #
          #  for (let s=0; s<vk_proof.nVars; s++) {
          #     for (let c in vk_proof.polsA[s]) {
          #           polA_T[c] = F.add(polA_T[c], F.mul(witness[s], vk_proof.polsA[s][c]));
          #     }
          #
          # Ex:
          # s iterates from 0 to 4
          # polsA = [{'1': 1}, {'2' : 1}, {'0': 3924283749832748327}, {}]
          # polsA[0] = {'1' : 1}, polsA[1] = {'2':1}, polsA[2] = {'0',3243243284}, polsA[3]={}
          # c in polsA[0] : '1', c in polsA[1] : '2', c in polsA[2] : '0', c in polsA[3] : {}
          # polA_T[1] = polA_t[1] + (witness[0] * polsA[0]['1'])
          # polA_T[2] = polA_t[2] + (witness[1] * polsA[1]['2'])
          # polA_T[0] = polA_t[0] + (witness[2] * polsA[2]['0'])
  
          polA_T = np.sum( np.multiply([1] + self.scl_array[:nVars], [polA_T] + self.pk['polsA'][:nVars]))
          polB_T = np.sum( np.multiply([1] + self.scl_array[:nVars], [polB_T] + self.pk['polsB'][:nVars]))
          polC_T = np.sum( np.multiply([1] + self.scl_array[:nVars], [polC_T] + self.pk['polsC'][:nVars]))
  
          polA_T = polA_T.expand_to_degree(nVars-1)
          polB_T = polB_T.expand_to_degree(nVars-1)
          polC_T = polC_T.expand_to_degree(nVars-1)
  
          # in : poly deg nVars-1.  out : poly deg nVars - 1
          polA_S = ZPoly(polA_T)
          polA_S.intt()
          # in : poly deg nVars-1.  out : poly deg nVars - 1
          polB_S = ZPoly(polB_T)
          polB_S.intt()
 
          polC_S = ZPoly(polC_T)
          polC_S.intt()

          # in : 2xpoly deg nVars-1.  out : poly deg nVars - 1
          polAB_S = ZPoly(polA_S)
          polAB_S.poly_mul(ZPoly(polB_S))

          polABC_S = polAB_S - polC_S
    
          inv_pol = None
    
          polZ_S = ZPoly([ZFieldElExt(-1)] + [ZFieldElExt(0) for i in range(m-1)] + [ZFieldElExt(1)])

          #return polABC_S
          polH_S = polABC_S.poly_div_snarks(polZ_S.get_degree())

          """
          H_S_copy = ZPoly(polH_S)
          H_S_copy.poly_mul(polZ_S)

          if H_S_copy == polABC_S:
            print "OK"
          else:
            print "KO"
          """

          # add coefficients of the polynomial (d2*A + d1*B - d3) + d1*d2*Z 

          polH_S = polH_S + d2 * polA_S + d2 * polB_S
          polH_S = polH_S.expand_to_degree(m)

          polH_S.zcoeff[0] -= d3
  
          # Z = x^m -1
          d1d2 = d1 * d2
          polH_S.zcoeff[m] += d1d2
          polH_S.zcoeff[0] -= d1d2
    
          polH_S = polH_S.norm()
  
          return polH_S

"""  
   t:
     ECPoints-EC_MAD_CUDA, ECPoints-EC2_MAD_CUDA, ECPoints-EC_MAD_CUDA, ECPoints-EC_MAD_CUDA, ECPoints-All
     H-d, H-ZPOLY_MADDM_H, H-IFFT_C, H-IFFT_A, H-IFFT_B, H-MUL, H-AB_C, H-DIV, 0,0,0, H-NORM, H-All
     END- MAD_CUDA, END-all
         
"""
