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
// File name  : groth_setup
//
// Date       : 13/05/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Zero Kowledge Groth setup implementation
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
import random
from subprocess import call
import logging
from multiprocessing import RawArray
from ctypes import c_uint32

from zutils import ZUtils
from zfield import *
from ecc import *
from zpoly import *
from constants import *
from cuda_wrapper import *
from pysnarks_utils import *
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

sys.path.append(os.path.abspath(os.path.dirname('../../config/')))
import cusnarks_config as cfg

class GrothSetup(object):
    

    def __init__(self, curve='BN128', in_circuit_f=None, out_circuit_f=None, out_circuit_format=FMT_MONT,
                 out_pk_f=None, out_vk_f=None, out_k_binformat=FMT_MONT, out_k_ecformat=EC_T_AFFINE, test_f=None,
                 benchmark_f=None, seed=None, snarkjs=None, keep_f=None, batch_size=20, reserved_cpus=0, write_table_f=None,
                 grouping=U256_BSELM):
 
        # Check valid folder exists
        if keep_f is None:
            print ("Repo directory needs to be provided\n")
            sys.exit(1)

        self.keep_f = gen_reponame(keep_f, sufix="_SETUP") 

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

        self.n_gpu = get_ngpu(max_used_percent=99.)
        self.grouping = grouping

        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group
        ZField(self.curve_data['prime'])
        # Initialize Field
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data)
        ZPoly.init(MOD_FIELD)

        ZField.set_field(MOD_FIELD)
        self.cir = getCircuit()

        self.cir_header   = None

        self.pk = getPK()

        self.t_S = {}

        self.ecbn128    = None 
        self.ec2bn128   = None 

        self.out_pk_f   = out_pk_f
        self.out_vk_f   = out_vk_f
        self.out_k_binformat = out_k_binformat
        self.out_k_ecformat = out_k_ecformat

        self.test_f = test_f
        self.toxic = {}
        self.snarkjs = snarkjs
 
        self.in_circuit_f = in_circuit_f
        self.out_circuit_f = out_circuit_f
        self.out_circuit_format = out_circuit_format
        self.in_circuit_format = FMT_EXT

        init_h()

        if test_f is None:
           self.test_f = test_f
        else:
           self.test_f= self.keep_f + '/' + test_f

        self.alfabeta_f = self.keep_f + '/' + 'alfabeta.json'
        if batch_size >  20:
          batch_size = 20
        self.batch_size = 1<<batch_size

        copy_input_files([in_circuit_f], self.keep_f)

        self.write_table_en = False
        self.write_table_f = write_table_f
        if write_table_f is not None:
          self.write_table_en = True

        self.sort_en = 0

        logging.info('#################################### ')
        logging.info('Staring Groth setup with the follwing arguments :')
        logging.info(' - curve : %s',curve)
        logging.info(' - in_circuit_f : %s', in_circuit_f)
        logging.info(' - out_circuit_f : %s', out_circuit_f)
        logging.info(' - out_circuit_format : %s', out_circuit_format)
        logging.info(' - out_pk_f : %s', out_pk_f)
        logging.info(' - out_vk_f : %s', out_vk_f)
        logging.info(' - out_k_binformat : %s', out_k_binformat)
        logging.info(' - out_k_ecformat : %s', out_k_ecformat)
        logging.info(' - test_f : %s', test_f)
        logging.info(' - benchmark_f : %s', benchmark_f)
        logging.info(' - seed : %s', seed)
        logging.info(' - snarkjs : %s', snarkjs)
        logging.info(' - keep_f : %s', keep_f)
        logging.info(' - bs : %s', batch_size)
        logging.info(' - sort enable : %s', self.sort_en)
        logging.info(' - write_table_en : %s', self.write_table_en)
        logging.info(' - table_f : %s', self.write_table_f)
        logging.info(' - grouping : %s', self.grouping)
        logging.info(' - n available GPUs : %s', self.n_gpu)
        logging.info('#################################### ')
        logging.info('')
        logging.info('')
         
        if self.in_circuit_f is not None:
           self.circuitRead()
           self.nbits = np.uint32(math.ceil(math.log(self.cir['nConstraints'] + \
                         self.cir['nPubInputs'] + \
                         self.cir['nOutputs'], 2)))

           self.roots_f = cfg.get_roots_file()
           self.n_bits_roots = cfg.get_n_roots()

           if self.n_bits_roots < self.nbits:
             logging.error('Insufficient number of roots in ' + self.roots_f + '. Required number of roots is '+ str(1<< self.nbits))
             sys.exit(1)
           
           self.roots_rdc_u256_sh = RawArray(c_uint32,  int((1 << self.nbits) * NWORDS_256BIT))
           self.roots_rdc_u256 = np.frombuffer(
                    self.roots_rdc_u256_sh,
                    dtype=np.uint32).reshape((1 << self.nbits, NWORDS_256BIT))
           np.copyto(self.roots_rdc_u256, readU256DataFile_h(self.roots_f.encode("UTF-8"), 1<<self.n_bits_roots, 1<<self.nbits) )

        else:
           logging.error ("Required input circuit")
           sys.exit(1)

    def __del__(self):
       release_h()
       logging.shutdown()

    def circuitRead(self):
        # cir Json to u256
        if self.in_circuit_f.endswith('.json'):
           self.cir = cirjson_to_vars(self.in_circuit_f, self.in_circuit_format,
                                      self.out_circuit_format)

           #u256 to bin
           if self.out_circuit_f is not None:
              cir_u256 = cirvars_to_bin(self.cir)
              writeU256DataFile_h(cir_u256, self.out_circuit_f.encode("UTF-8"))

        elif self.in_circuit_f.endswith('.bin'):
             cir_u256 = readU256CircuitFile_h(self.in_circuit_f.encode("UTF-8"))
             self.cir = cirbin_to_vars(cir_u256)

        elif self.in_circuit_f.endswith('.r1cs'):
             r1cs_header, r1csa, r1csb, r1csc = readR1CSFile_h(self.in_circuit_f.encode("UTF-8"))
             self.cir = cirr1cs_to_vars(r1cs_header, r1csa, r1csb, r1csc)
           

        logging.info('############################################## ')
        logging.info(' - NVars        : %s',int(self.cir['nVars']))
        logging.info(' - nOutputs     : %s',int(self.cir['nOutputs']))
        logging.info(' - nConstraints : %s',int(self.cir['nConstraints']))
        logging.info('############################################## ')

    def launch_snarkjs(self, mode):
        snarkjs = {}
        if mode=="setup" or mode=="alfabeta_12":
          # snarkjs setup is launched with circuit.json, format extended. Convert input file if necessary
          if self.in_circuit_f.endswith('.json') and self.in_circuit_format == FMT_EXT :
             circuit_file = self.in_circuit_f
          elif mode == "alfabeta_12":
             circuit_file = self.in_circuit_f
          else:
             logging.error("To launch snarkjs, input circuit %s needs to be a json file and format %s cannot be Montgomery", self.in_circuit_f, self.in_circuit_format)
             return
        
          snarkjs['pk_f'] = self.keep_f + '/' + 'tmp_pk_f.json'
          snarkjs['vk_f'] = self.keep_f + '/' + 'tmp_vk_f.json'

          if mode=="alfabeta_12":
            alfabeta_command = "--fs"
            alfabeta_file    = self.alfabeta_f
          else:
            alfabeta_command = ""
            alfabeta_file    = ""

          if mode=="setup" and self.test_f is not None:
             toxic_command = "--d"
             toxic_file = self.test_f
          else:
             toxic_command = ""
             toxic_file = ""
       
          call([self.snarkjs, "setup", "-c", circuit_file, "--pk", snarkjs['pk_f'], "--vk", snarkjs['vk_f'], "--protocol", "groth", alfabeta_command, alfabeta_file, toxic_command, toxic_file])
       
        return snarkjs
   
    def setup(self):
        logging.info('#################################### ')
        logging.info('')
        logging.info('')
        logging.info(' Starting setup....')

        ZField.set_field(MOD_FIELD)
        start_s = time.time()
        
        #Init PK
        cirvars_to_pkvars(self.pk, self.cir)
        self.pk['k_binformat'] = self.out_k_binformat
        self.pk['k_ecformat'] =  self.out_k_ecformat


        prime = ZField.get_extended_p()

        self.toxic['t'] = ZFieldElExt(random.randint(1,prime.as_long()-1))

        self.calculatePoly()
        end = time.time()
        self.t_S['cal Poly'] = end - start_s

        start = time.time()
        self.calculateEncryptedValuesAtT()

        end_s = time.time()
        self.t_S['total'] = end_s - start_s
        self.t_S['cal Crypto'] = end_s - start

        logging.info('')
        logging.info('#################################### ')
        logging.info('')
        logging.info('EC P Density')
        logging.info('A      : %s',round(self.pk['A_density'],2))
        logging.info('B1     : %s',round(self.pk['B1_density'],2))
        logging.info('B2     : %s',round(self.pk['B2_density'],2))
        logging.info('C      : %s',round(self.pk['C_density'],2))
        logging.info('hExps  : %s',round(self.pk['hExps_density'],2))
        logging.info('')
        logging.info('#################################### ')

        logging.info('')
        logging.info('Setup completed')
        logging.info('')
        logging.info('#################################### ')
        logging.info('')
        logging.info('')

        self.logTimeResults()

        # TODO : The idea is to do some precalculations to compute multiexp later during
        # proof. However, the way is is laid out it takes to much space. I comment this part
        # for now until I come up with a better way
        if self.write_table_en:
          self.write_tables(all_tables=1)

        self.write_pk()
        self.write_vk()

        copy_input_files([self.out_vk_f, self.out_pk_f, self.out_circuit_f],self.keep_f)
        self.test_results()

        return
   
    def create_correct_pkjson(self): 
          #Get pk json
          test_pk_f = self.keep_f + self.out_pk_f
          test_vk_f = self.keep_f + self.out_vk_f
          if self.out_pk_f.endswith('.bin')  or  \
              self.out_k_binformat != FMT_EXT or \
              self.out_k_ecformat != EC_T_AFFINE:
            if self.out_pk_f.endswith('.bin') :
              test_pk_f = test_pk_f.replace('.bin','_cpy.json') 
            else :
              test_pk_f = test_pk_f.replace('.json','_cpy.json') 
            pk_dict = pkvars_to_json(FMT_EXT,EC_T_AFFINE, self.pk)
            pk_json = json.dumps(pk_dict, indent=4, sort_keys=True)
            #if os.path.exists(test_pk_f):
              #os.remove(test_pk_f)
            f = open(test_pk_f, 'w')
            print(pk_json, file=f)
            f.close()

          if self.out_vk_f.endswith('.bin') :
          #Get vk json
            test_vk_f = test_vk_f.replace('bin','json')
            vk_dict = self.vars_to_vkdict()
            vk_json = json.dumps(vk_dict, indent=4, sort_keys=True)
            #if os.path.exists(test_vk_f):
              #os.remove(test_vk_f)
            f = open(test_vk_f, 'w')
            print(vk_json, file=f)
            f.close()

          return test_pk_f, test_vk_f

    def test_results(self):
        logging.info('#################################### ')
        logging.info('')
        logging.info('')
        logging.info('Checking results...')

        if self.test_f is not None:
          # Write toxic json
          toxic_dict={}
          toxic_dict['t'] = str(self.toxic['t'].as_long())
          toxic_dict['kalfa'] = str(self.toxic['kalfa'].as_long())   
          toxic_dict['kbeta'] = str(self.toxic['kbeta'].as_long())
          toxic_dict['kdelta'] = str(self.toxic['kdelta'].as_long())
          toxic_dict['kgamma'] = str(self.toxic['kgamma'].as_long())
          toxic_json = json.dumps(toxic_dict, indent=4, sort_keys=True)
          #if os.path.exists(self.test_f):
          #    os.remove(self.test_f)
          f = open(self.test_f, 'w')
          print(toxic_json, file=f)
          f.close()

          test_pk_f, test_vk_f = self.create_correct_pkjson()

       
          # Launch snarkjs
          logging.info('Launching snarkjs setup to compare result...')
          snarkjs = self.launch_snarkjs("setup")


          worker = mp.Pool(processes=min(2,mp.cpu_count()))

          # Compare results
          r1 = worker.apply_async(pysnarks_compare, args=(test_pk_f, snarkjs['pk_f'], ['A', 'B1', 'B2', 'C', 'hExps', 'polsA', 'polsB',
                                                                   'vk_alfa_1', 'vk_beta_1', 'vk_beta_2',
                                                                   'vk_delta_1', 'vk_delta_2'], self.pk['nPublic'] ))
          r2 = worker.apply_async(pysnarks_compare, args=(test_vk_f, snarkjs['vk_f'], ['IC', 'vk_alfa_1', 'vk_alfabeta_12', 'vk_beta_2', 
                                                                   'vk_delta_2', 'vk_gamma_2'],0))

          pk_r = r1.get()
          vk_r = r2.get()

          worker.terminate()
          if pk_r:
            logging.info('Proving Key matches')
          else:
            logging.info('Proving Key failed')
          if vk_r:
            logging.info('Verification Key matches')
          else:
            logging.info('Verification Key failed')

          if pk_t and vk_r:
            logging.info('Setup passed')
          else:
            logging.info('Setup failed')

          return pk_r and vk_r
        else:
          logging.info('Results not verified')
          return True


    def calculatePoly(self):
        logging.info(' Starting calculatePoly')
        self.computeHeader()

        logging.info(' Starting polsA')
        self.pk['polsA'] = cirr1cs_to_mpoly(self.cir['R1CSA'], self.cir_header, self.cir['cirformat'], 1)
        self.pk['polsA_nWords'] = np.uint32(self.pk['polsA'].shape[0])
        del self.cir['R1CSA']

        logging.info(' Starting polsB')
        self.pk['polsB'] = cirr1cs_to_mpoly(self.cir['R1CSB'], self.cir_header, self.cir['cirformat'], 0)
        self.pk['polsB_nWords'] = np.uint32(self.pk['polsB'].shape[0])
        del self.cir['R1CSB']

        logging.info(' Starting polsC')
        self.pk['polsC'] = cirr1cs_to_mpoly(self.cir['R1CSC'], self.cir_header, self.cir['cirformat'], 0)
        self.pk['polsC_nWords'] = np.uint32(self.pk['polsC'].shape[0])
        del self.cir

        return

    def evalLagrangePoly(self, bits):
       """
        m : int
        t : ZFieldElRedc
       """
       logging.info(' Starting evalLagrangePoly')
       m = 1 << bits
       t_rdc = self.toxic['t'].reduce()
       tm = (t_rdc ** int(m))
       u_u256 = np.zeros((m,NWORDS_256BIT),dtype=np.uint32)
       trdc_u256 = t_rdc.as_uint256()
      
       z = tm.extend() - 1
       z_rdc = z.reduce()
       if tm == ZFieldElExt(1).reduce():
         for i in xrange(m): 
           #TODO : roots[0] is always 1. Does this make any sense? check javascript version
           if self.roots_rdc_u256[0] == trdc_u256:
             u_u256[i] = ZFieldElExt(1).reduce().as_uint256()
             return z.as_uint256(), u_u256

       l_rdc = z_rdc * ZFieldElExt(int(m)).inv().reduce()
       lrdc_u256 = l_rdc.as_uint256()

       pidx = ZField.get_field()
       u_u256 = evalLagrangePoly_h(trdc_u256,lrdc_u256, self.roots_rdc_u256, pidx)

       logging.info(' Deleting roots from memory')
       del self.roots_rdc_u256

       return z.as_uint256(), u_u256
   
    def calculateEncryptedValuesAtT(self):
      start = time.time()
      a_t_u256, b_t_u256, c_t_u256, z_t_u256 = self.calculateValuesAtT()
      end = time.time()
      self.t_S['cal Val'] = end - start

      start = time.time()
      prime = ZField.get_extended_p()
      curve_params = self.curve_data['curve_params']
      curve_params_g2 = self.curve_data['curve_params_g2']

      # Tocix k extended
      self.toxic['kalfa'] = ZFieldElExt(random.randint(1,prime.as_long()-1))
      self.toxic['kbeta'] = ZFieldElExt(random.randint(1,prime.as_long()-1))
      self.toxic['kdelta'] = ZFieldElExt(random.randint(1,prime.as_long()-1))
      self.toxic['kgamma'] = ZFieldElExt(random.randint(1,prime.as_long()-1))

      toxic_invDelta = self.toxic['kdelta'].inv()
      toxic_invGamma = self.toxic['kgamma'].inv()

      ZField.set_field(MOD_GROUP)
      Gx = ZFieldElExt(curve_params['Gx'])
      Gy = ZFieldElExt(curve_params['Gy'])
      G2x = Z2FieldEl([curve_params_g2['Gx1'], curve_params_g2['Gx2']])
      G2y = Z2FieldEl([curve_params_g2['Gy1'], curve_params_g2['Gy2']])

      G1 = ECCJacobian([Gx,Gy]).reduce()
      G2 = ECCJacobian([G2x, G2y]).reduce()

      # vk coeff MONT
      logging.info(' Computing alfas')
    
      self.pk['alfa_1'] = ec_jac2aff_h(
                 np.reshape(
                     ec_jacscmul_h(
                              self.toxic['kalfa'].as_uint256().reshape(-1), 
                              G1.as_uint256(G1).reshape(-1),
                              ZField.get_field(),
                              0),
                     -1),
                 ZField.get_field(),
                 1)
      self.pk['beta_1'] = ec_jac2aff_h(
                 np.reshape(
                     ec_jacscmul_h(
                              self.toxic['kbeta'].as_uint256().reshape(-1), 
                              G1.as_uint256(G1).reshape(-1),
                              ZField.get_field(),
                              0),
                     -1),
                 ZField.get_field(),
                 1)
      self.pk['delta_1'] = ec_jac2aff_h(
                 np.reshape(
                     ec_jacscmul_h(
                              self.toxic['kdelta'].as_uint256().reshape(-1), 
                              G1.as_uint256(G1).reshape(-1),
                              ZField.get_field(),
                              0),
                     -1),
                 ZField.get_field(),
                 1)
    
      self.pk['beta_2'] = ec2_jac2aff_h(
                 np.reshape(
                     ec2_jacscmul_h(
                              self.toxic['kbeta'].as_uint256().reshape(-1), 
                              G2.as_uint256(G2).reshape(-1),
                              ZField.get_field(),
                              0),
                     -1),
                 ZField.get_field(),
                 1)
      self.pk['delta_2'] = ec2_jac2aff_h(
                 np.reshape(
                     ec2_jacscmul_h(
                              self.toxic['kdelta'].as_uint256().reshape(-1), 
                              G2.as_uint256(G2).reshape(-1),
                              ZField.get_field(),
                              0),
                     -1),
                 ZField.get_field(),
                 1)
      self.pk['gamma_2'] = ec2_jac2aff_h(
                 np.reshape(
                     ec2_jacscmul_h(
                              self.toxic['kgamma'].as_uint256().reshape(-1), 
                              G2.as_uint256(G2).reshape(-1),
                              ZField.get_field(),
                              0),
                     -1),
                 ZField.get_field(),
                 1)

      if self.n_gpu :
        self.ecbn128    =  ECBN128(self.batch_size,seed=self.seed)
        self.ec2bn128    = EC2BN128(self.batch_size,seed=self.seed)
      else :
        self.ecbn128    =  None
        self.ec2bn128    = None

      logging.info(' Computing EC Point A')
      end = time.time()
      self.t_S['init k'] = end - start
      start = time.time()
      # a_t, b_t and c_t are in ext
      sorted_idx = sortu256_idx_h(a_t_u256,self.sort_en)
      ecbn128_samples = np.concatenate((a_t_u256[sorted_idx],G1.as_uint256(G1)[:2]))
      self.pk['A'],t1 = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field(), batch_size=self.batch_size)
      logging.info(' Converting EC Point A to Affine coordinates')
      self.pk['A'] = ec_jac2aff_h(self.pk['A'].reshape(-1),ZField.get_field(),1)
      self.assert_isoncurve('A', samples= ecbn128_samples)

      unsorted_idx = np.argsort(sorted_idx)
      self.pk['A'] = np.reshape(self.pk['A'],(-1,2,NWORDS_256BIT))[unsorted_idx]
      self.pk['A'] = np.reshape(self.pk['A'],(-1,NWORDS_256BIT))
      self.pk['A_nWords'] = np.uint32(self.pk['A'].shape[0] * NWORDS_256BIT )
      infv = ec_isinf(np.reshape(self.pk['A'],-1), ZField.get_field())
      self.pk['A_density'] = 100.0 - 100.0 *  np.count_nonzero(infv == 1) / len(infv)

      end = time.time()
      self.t_S['A'] = end - start

      self.t_S['A gpu'] = t1
      start = time.time()

    
      logging.info(' Computing EC Point B1')
      sorted_idx = sortu256_idx_h(b_t_u256,self.sort_en)
      ecbn128_samples[:-2] = b_t_u256[sorted_idx]
      self.pk['B1'],t1 = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field(), batch_size=self.batch_size)
      logging.info(' Converting EC Point B1 to Affine coordinates')
      self.pk['B1'] = ec_jac2aff_h(self.pk['B1'].reshape(-1),ZField.get_field(),1)
      self.assert_isoncurve('B1', samples=ecbn128_samples)

      unsorted_idx = np.argsort(sorted_idx)
      self.pk['B1'] = np.reshape(self.pk['B1'],(-1,2,NWORDS_256BIT))[unsorted_idx]
      self.pk['B1']=np.reshape(self.pk['B1'],(-1,NWORDS_256BIT))
      self.pk['B1_nWords'] = np.uint32(self.pk['B1'].shape[0] * NWORDS_256BIT)

      infv = ec_isinf(np.reshape(self.pk['B1'],-1), ZField.get_field())
      self.pk['B1_density'] = 100.0  - 100.0 *  np.count_nonzero(infv == 1) / len(infv)

      end= time.time()
      self.t_S['B1 gpu'] = t1
      self.t_S['B1'] = end-start

      logging.info(' Computing EC Point B2')
      start= time.time()
      ec2bn128_samples = np.concatenate((b_t_u256[sorted_idx],G2.as_uint256(G2)[:4]))
      #ec2bn128_samples = np.concatenate((b_t_u256,G2.as_uint256(G2)[:4]))
      self.pk['B2'],t1 = ec_sc1mul_cuda(self.ec2bn128, ec2bn128_samples, ZField.get_field(), ec2=True, batch_size=self.batch_size)
      logging.info(' Converting EC Point B2 to Affine coordinates')
      self.pk['B2'] = ec2_jac2aff_h(self.pk['B2'].reshape(-1),ZField.get_field(),1)
      self.assert_isoncurve('B2', samples = ec2bn128_samples)
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['B2'] = np.reshape(self.pk['B2'],(-1,4,NWORDS_256BIT))[unsorted_idx]
      #self.B2 = np.reshape(self.B2,(-1,6,NWORDS_256BIT))
      self.pk['B2'] = np.reshape(self.pk['B2'],(-1,NWORDS_256BIT))
      #ECC.from_uint256(self.B2.reshape((-1,2,8))[0:3],reduced=True, in_ectype=2, out_ectype=2,ec2=True)[0].extend().as_list()
      self.pk['B2_nWords'] = np.uint32(self.pk['B2'].shape[0] * NWORDS_256BIT)

      infv = ec2_isinf(np.reshape(self.pk['B2'],-1), ZField.get_field())
      self.pk['B2_density'] = 100.0 - 100 * np.count_nonzero(infv == 1) / len(infv)

      self.t_S['B2 gpu'] = t1
      end= time.time()
      self.t_S['B2'] = end - start
      del ec2bn128_samples

      start = time.time()

      logging.info(' Computing EC Point C')
      ZField.set_field(MOD_FIELD)
      pidx = ZField.get_field()
      ps_u256 = GrothSetupComputePS_h(self.toxic['kalfa'].reduce().as_uint256(), self.toxic['kbeta'].reduce().as_uint256(),
                                      toxic_invDelta.reduce().as_uint256(),
                                      a_t_u256, b_t_u256, c_t_u256, self.pk['nPublic']+1, self.pk['nVars'], pidx )

      ZField.set_field(MOD_GROUP)
      sorted_idx = sortu256_idx_h(ps_u256,self.sort_en)
      ecbn128_samples = np.concatenate((ps_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.pk['C'],t1 = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field(), batch_size=self.batch_size)
      logging.info(' Converting EC Point C to Affine coordinates')
      self.pk['C'] = ec_jac2aff_h(self.pk['C'].reshape(-1),ZField.get_field(),1)
      self.assert_isoncurve('C', samples=ecbn128_samples)
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['C'] = np.reshape(self.pk['C'],(-1,2,NWORDS_256BIT))[unsorted_idx]
      #ECC.from_uint256(self.C[12],reduced=True, in_ectype=2, out_ectype=2)[0].extend().as_list()
      self.pk['C']=np.concatenate((np.zeros(((int(self.pk['nPublic']+1)*2),NWORDS_256BIT),dtype=np.uint32),np.reshape(self.pk['C'],(-1,NWORDS_256BIT))))
      self.pk['C_nWords'] = np.uint32(self.pk['C'].shape[0] * NWORDS_256BIT)

      infv = ec_isinf(np.reshape(self.pk['C'],-1), ZField.get_field())
      self.pk['C_density'] = 100.0 - 100.0 * np.count_nonzero(infv == 1) / len(infv)

      del ps_u256

      self.t_S['C gpu'] = t1
      end = time.time()
      self.t_S['C'] =end - start 

      start = time.time()

      logging.info(' Computing EC Point hExps')
      maxH = self.pk['domainSize']+1
      self.pk['hExps'] = np.zeros((maxH,NWORDS_256BIT),dtype=np.uint32)

      ZField.set_field(MOD_FIELD)
      pidx = ZField.get_field()
      zod_u256 = montmultN_h(toxic_invDelta.reduce().as_uint256(), z_t_u256, pidx)
      eT_u256 = GrothSetupComputeeT_h(self.toxic['t'].reduce().as_uint256(), np.reshape(zod_u256,-1), maxH, pidx)

      ZField.set_field(MOD_GROUP)
      sorted_idx = sortu256_idx_h(eT_u256,self.sort_en)
      ecbn128_samples = np.concatenate((eT_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.pk['hExps'],t1 = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field(), batch_size=self.batch_size)
      logging.info(' Converting EC Point hExps to Affine coordinates')
      self.pk['hExps'] = ec_jac2aff_h(self.pk['hExps'].reshape(-1),ZField.get_field(),1)
      self.assert_isoncurve('hExps', samples=ecbn128_samples)
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['hExps'] = np.reshape(self.pk['hExps'],(-1,2,NWORDS_256BIT))[unsorted_idx]
      self.pk['hExps']=np.reshape(self.pk['hExps'],(-1,NWORDS_256BIT))
      self.pk['hExps_nWords'] = np.uint32(self.pk['hExps'].shape[0] * NWORDS_256BIT)

      end = time.time()
      self.t_S['hExps'] =end - start 
      self.t_S['hExps gpu'] = t1

      infv = ec_isinf(np.reshape(self.pk['hExps'],-1), ZField.get_field())
      self.pk['hExps_density'] = 100.0 - 100.0 *  np.count_nonzero(infv == 1) / len(infv)

      del eT_u256

      start = time.time()

      logging.info(' Computing EC Point IC')
      ZField.set_field(MOD_FIELD)
      pidx = ZField.get_field()
      ps_u256 = GrothSetupComputePS_h(self.toxic['kalfa'].reduce().as_uint256(), self.toxic['kbeta'].reduce().as_uint256(),
                                      toxic_invGamma.reduce().as_uint256(),
                                      a_t_u256, b_t_u256, c_t_u256, 0, self.pk['nPublic']+1, pidx )
      ZField.set_field(MOD_GROUP)
      sorted_idx = sortu256_idx_h(ps_u256,self.sort_en)
      ecbn128_samples = np.concatenate((ps_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.pk['IC'],t1 = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field(), batch_size=self.batch_size)
      logging.info(' Converting EC Point IC to Affine coordinates')
      self.pk['IC'] = ec_jac2aff_h(self.pk['IC'].reshape(-1),ZField.get_field(),1)
      self.assert_isoncurve('IC', samples = ecbn128_samples)
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['IC'] = np.reshape(self.pk['IC'],(-1,2,NWORDS_256BIT))[unsorted_idx]
      self.pk['IC'] = np.uint32(np.reshape(self.pk['IC'],(-1,NWORDS_256BIT)))

      end = time.time()
      self.t_S['IC'] =end - start 
      self.t_S['IC gpu'] = t1

    def assert_isoncurve(self, ec_str, max_fails=5, samples=None):
      if self.test_f is None:
          return

      dim2=2
      ec2=0
      if ec_str=='B2':
         dim2=4
         ec2=1

      logging.info(' Checking EC Point belongs to curve')
      pok = np.asarray([ec_isoncurve_h(np.reshape(x,-1), 1, ec2, 
                    ZField.get_field()) for x in np.reshape(self.pk[ec_str],(-1,dim2,NWORDS_256BIT))])
      if not all(pok > 0):
        n_fails = np.asarray(np.where(pok==0)[0]).shape[0]
        logging.error("%s containts %s points not on curve: \n",ec_str, n_fails)
        fail_idx = np.asarray(np.where(pok==0)[0])
        logging.error("indexes - %s\n", np.asarray(np.where(pok==0)[0])[:max_fails])
        ecp = np.reshape(self.pk[ec_str],(-1, dim2, NWORDS_256BIT))
        #writeU256DataFile_h(np.reshape(samples,-1), '/local/david/iden3/cusnarks/test/c/aux_data/samples.bin'.encode("UTF-8"))
        #writeU256DataFile_h(np.reshape(self.pk[ec_str],-1),'/local/david/iden3/cusnarks/test/c/aux_data/ecp.bin'.encode("UTF-8"))
 
        logging.error("Input scalars : %s\n", samples[fail_idx])
        logging.error("G(affine/Mont) : %s\n", samples[-dim2:])
        logging.error("Points not on curve(Jac/Mont) : %s\n", ecp[fail_idx])
        sys.exit(1)

    def calculateValuesAtT(self):
       # Required z_t es Ext, polsA/B/C are in Mont format
       # u is Mont
       z_t_u256, u = self.evalLagrangePoly(self.pk['domainBits'])

       logging.info(' Starting mpoly_madd_h')
       pidx = ZField.get_field()

       logging.info(' Starting computing a_t_u256')
       a_t_u256 = mpoly_madd_h(self.pk['polsA'], u.reshape(-1), self.pk['nVars'], pidx)
       logging.info(' Starting computing b_t_u256')
       b_t_u256 = mpoly_madd_h(self.pk['polsB'], u.reshape(-1), self.pk['nVars'], pidx)
       logging.info(' Starting computing c_t_u256')
       c_t_u256 = mpoly_madd_h(self.pk['polsC'], u.reshape(-1), self.pk['nVars'], pidx)

       # a,b,c are Ext
       return a_t_u256, b_t_u256, c_t_u256, z_t_u256


    def computeHeader(self):

        self.cir_header = {'nWords' : self.cir['nWords'],
                  'nPubInputs' : self.cir['nPubInputs'],
                  'nOutputs' : self.cir['nOutputs'],
                  'nVars' : self.cir['nVars'],
                  'nConstraints' : self.cir['nConstraints'],
                  'cirformat' : self.cir['cirformat'],
                  'R1CSA_nWords' : self.cir['R1CSA_nWords'],
                  'R1CSB_nWords' : self.cir['R1CSB_nWords'],
                  'R1CSC_nWords' : self.cir['R1CSC_nWords']}

    def write_tables(self, all_tables=1):
       logging.info('#################################### ')
       logging.info('')
       logging.info('Writing Table files')

       domainSize   =  int(self.pk['domainSize'])
       nPublic =  int(self.pk['nPublic'])

       nWords_offset = ECTABLE_DATA_OFFSET_WORDS
       nWords_offset_dw = dw2w(nWords_offset)

       if all_tables:
         nTables_A = int((self.pk['A'].shape[0] / ECP_JAC_INDIMS + 2 + self.grouping - 1)/self.grouping) 
         nWords1_A = (nTables_A << self.grouping ) * NWORDS_256BIT * ECP_JAC_INDIMS + nWords_offset
         nWords1_A_dw = dw2w(nWords1_A)
         nTables_B2 = int((self.pk['B2'].shape[0] / ECP2_JAC_INDIMS + 2 + self.grouping - 1)/self.grouping) 
         nWords1_B2 = (nTables_B2 << self.grouping ) * NWORDS_256BIT * ECP2_JAC_INDIMS + nWords1_A
         nWords1_B2_dw = dw2w(nWords1_B2)
         nTables_B1 = int((self.pk['B1'].shape[0] / ECP_JAC_INDIMS + 2 + self.grouping - 1)/self.grouping)
         nWords1_B1 = (nTables_B1 << self.grouping ) * NWORDS_256BIT * ECP_JAC_INDIMS + nWords1_B2
         nWords1_B1_dw = dw2w(nWords1_B1)
         nTables_C = int((self.pk['C'][2*(nPublic+1):].shape[0] / ECP_JAC_INDIMS + self.grouping - 1)/self.grouping)
         nWords1_C = (nTables_C << self.grouping ) * NWORDS_256BIT * ECP_JAC_INDIMS + nWords1_B1
         nWords1_C_dw = dw2w(nWords1_C)
       else:
         nWords1_A_dw = dw2w(nWords_offset)
         nWords1_B2_dw = dw2w(nWords_offset)
         nWords1_B1_dw = dw2w(nWords_offset)
         nWords1_C_dw = dw2w(nWords_offset)
         nTables_A = 0
         nTables_B2 = 0
         nTables_B1 = 0
         nTables_C = 0
         nWords1_C = 0

       nTables_hExps = int((domainSize + self.grouping - 1)/self.grouping) 
       nWords1_hExps = (nTables_hExps << self.grouping ) * NWORDS_256BIT * ECP_JAC_INDIMS + nWords1_C
       nWords1_hExps_dw = dw2w(nWords1_hExps)

       nWords = np.concatenate(([np.uint32(self.grouping)], nWords_offset_dw, 
                                nWords1_A_dw,  
                                nWords1_B2_dw, 
                                nWords1_B1_dw, 
                                nWords1_C_dw,  nWords1_hExps_dw))

       writeU256DataFile_h(nWords, self.write_table_f.encode("UTF-8"))
       write_group_size = 1000

       if all_tables == 1:
         logging.info(' Computing EC Point A Tables')
         super_group =  np.concatenate((
                                             self.pk['A'],
                                             self.pk['alfa_1'],
                                             self.pk['delta_1'] 
                                     ))

         groups = np.arange(0,super_group.shape[0], self.grouping*write_group_size*ECP_JAC_INDIMS) 
         groups = np.append(groups, len(super_group)+1)
         for gidx in range(len(groups)-1):
           table = ec_inittable_h(
                                 np.reshape(super_group[groups[gidx]:groups[gidx+1]],
                                             -1), self.grouping, MOD_GROUP, 1)
           table = ec_jac2aff_h(np.reshape(table,-1),MOD_GROUP,1)
           appendU256DataFile_h(np.reshape(table,-1), self.write_table_f.encode("UTF-8"))
         
         logging.info(' Done computing EC Point A Tables')
  
         logging.info(' Computing EC Point B2 Tables')
         super_group =  np.concatenate((
                                             self.pk['B2'],
                                             self.pk['beta_2'],
                                             self.pk['delta_2'] 
                                     ))

         groups = np.arange(0,super_group.shape[0], self.grouping*write_group_size*ECP2_JAC_INDIMS) 
         groups = np.append(groups, len(super_group)+1)
         for gidx in range(len(groups)-1):
           table = ec2_inittable_h(
                              np.reshape(super_group[groups[gidx]:groups[gidx+1]],
                                   -1), self.grouping, MOD_GROUP, 1)
           table = ec2_jac2aff_h(np.reshape(table,-1),MOD_GROUP,1)
           appendU256DataFile_h(np.reshape(table,-1), self.write_table_f.encode("UTF-8"))
         logging.info(' Done computing EC Point B2 Tables')

         logging.info(' Computing EC Point B1 Tables')
         super_group =  np.concatenate((
                                             self.pk['B1'],
                                             self.pk['beta_1'],
                                             self.pk['delta_1'] 
                                     ))

         groups = np.arange(0,super_group.shape[0], self.grouping*write_group_size*ECP_JAC_INDIMS) 
         groups = np.append(groups, len(super_group)+1)
         for gidx in range(len(groups)-1):
           table = ec_inittable_h(
                                 np.reshape(super_group[groups[gidx]:groups[gidx+1]],
                                             -1), self.grouping, MOD_GROUP, 1)
           table = ec_jac2aff_h(np.reshape(table,-1),MOD_GROUP,1)
           appendU256DataFile_h(np.reshape(table,-1), self.write_table_f.encode("UTF-8"))
         logging.info(' Done computing EC Point B1 Tables')

         logging.info(' Computing EC Point C Tables')
         super_group =  self.pk['C'][2*(nPublic+1):]

         groups = np.arange(0,super_group.shape[0], self.grouping*write_group_size*ECP_JAC_INDIMS) 
         groups = np.append(groups, len(super_group)+1)
         for gidx in range(len(groups)-1):
           table = ec_inittable_h(
                                 np.reshape(super_group[groups[gidx]:groups[gidx+1]],
                                             -1), self.grouping, MOD_GROUP, 1)
           table = ec_jac2aff_h(np.reshape(table,-1),MOD_GROUP,1)
           appendU256DataFile_h(np.reshape(table,-1), self.write_table_f.encode("UTF-8"))
         logging.info(' Done computing EC Point C Tables')

       logging.info(' Computing EC Point hExps Tables')
       super_group =  np.concatenate((
                                             self.pk['hExps'][:2*(domainSize-1)],
                                             self.pk['delta_1'] 
                                     ))

       groups = np.arange(0,super_group.shape[0], self.grouping*write_group_size*ECP_JAC_INDIMS) 
       groups = np.append(groups, len(super_group)+1)
       for gidx in range(len(groups)-1):
           table = ec_inittable_h(
                                 np.reshape(super_group[groups[gidx]:groups[gidx+1]],
                                             -1), self.grouping, MOD_GROUP, 1)
           table = ec_jac2aff_h(np.reshape(table,-1),MOD_GROUP,1)
           appendU256DataFile_h(np.reshape(table,-1), self.write_table_f.encode("UTF-8"))
         
       logging.info(' Done computing EC Point hExps Tables')

       logging.info('')
       if all_tables:
         logging.info('Table1 A     : %s elements', nTables_A)
         logging.info('Table1 B2    : %s elements', nTables_B2)
         logging.info('Table1 B1    : %s elements', nTables_B1)
         logging.info('Table1 C     : %s elements', nTables_C)
       logging.info('Table1 hExps : %s elements', nTables_hExps)


       logging.info('')
       logging.info('')
       logging.info('#################################### ')
     
    def write_pk(self):

       logging.info('#################################### ')
       logging.info('')
       logging.info('Writing Proving Key')
       if self.out_pk_f.endswith('.json') :
         pk_dict = pkvars_to_json(self.out_k_binformat, self.out_k_ecformat,self.pk)
         pk_json = json.dumps(pk_dict, indent=4, sort_keys=True)
         #if os.path.exists(self.out_pk_f):
         #    os.remove(self.out_pk_f)
         f = open(self.out_pk_f, 'w')
         print(pk_json, file=f)
         f.close()

       elif self.out_pk_f.endswith('bin') :
         pkvars_to_file(self.out_k_binformat, self.out_k_ecformat, self.pk, self.out_pk_f, ext=True)

       else :
         logging.info ("No valid proving key file  %s provided", self.out_pk_f)
         sys.exit(1)
       logging.info('')
       logging.info('#################################### ')
       logging.info('')
        
    def write_vk(self):
       logging.info('#################################### ')
       logging.info('')
       logging.info('')
       logging.info('Writing Verification Key')
       #Fill alfa and beta values and call snarkjs to compute vk value
       vk_dict = self.vars_to_vkdict(alfabeta=True)
       vk_json = json.dumps(vk_dict, indent=4, sort_keys=True)
       #if os.path.exists(self.alfabeta_f):
       #     os.remove(self.alfabeta_f)
       f = open(self.alfabeta_f, 'w')
       print(vk_json, file=f)
       f.close()

       logging.info('Calling snarkjs to compute pairing')
       self.launch_snarkjs("alfabeta_12")

       if self.out_vk_f.endswith('.json') :
         vk_dict = self.vars_to_vkdict()
         vk_json = json.dumps(vk_dict, indent=4, sort_keys=True)
         #if os.path.exists(self.out_vk_f):
         #   os.remove(self.out_vk_f)
         f = open(self.out_vk_f, 'w')
         print(vk_json, file=f)
         f.close()

       elif self.out_vk_f.endswith('bin') :
         vk_bin = self.vars_to_vkbin()

         if vk_bin is not None:
            writeU256DataFile_h(vk_bin, self.out_vk_f.encode("UTF-8"))
       logging.info('')
       logging.info('')
       logging.info('#################################### ')

    def vars_to_vkdict(self, alfabeta=False):
      # TODO : only suported formats for vk are .json, affine and extended 
      vk_dict = {}
      ZField.set_field(MOD_GROUP)
      vk_dict['vk_alfa_1'] = ECC.from_uint256(
                 self.pk['alfa_1'],
                 in_ectype=EC_T_AFFINE,
                 out_ectype=EC_T_AFFINE,
                 reduced=True,
                 remove_last=True)[0].extend().as_str()
      vk_dict['vk_beta_2'] = ECC.from_uint256(
                 self.pk['beta_2'].reshape((-1,2,NWORDS_256BIT)),
                 in_ectype=EC_T_AFFINE,
                 out_ectype=EC_T_AFFINE,
                 reduced=True,
                 ec2=True,
                 remove_last=True)[0].extend().as_str()

      if not alfabeta :
        f = open(self.alfabeta_f,'r')
        vk_dict['vk_alfabeta_12'] = json.load(f)
        f.close()
        vk_dict['protocol'] = "groth"
        vk_dict['field_r'] = str(ZFieldElExt.from_uint256(self.pk['field_r']).as_long())
        vk_dict['group_q'] = str(ZFieldElExt.from_uint256(self.pk['group_q']).as_long())
        vk_dict['binFormat'] = "normal"

        vk_dict['Rbitlen'] = int(self.pk['Rbitlen'])
        vk_dict['ecFormat'] = "affine"

        vk_dict['nVars'] = int(self.pk['nVars'])
        vk_dict['nPublic'] = int(self.pk['nPublic'])
        vk_dict['domainBits'] = int(self.pk['domainBits'])
        vk_dict['domainSize'] = int(self.pk['domainSize'])
  
        vk_dict['vk_delta_2'] = ECC.from_uint256(
                   self.pk['delta_2'].reshape((-1,2,NWORDS_256BIT)),
                   in_ectype=EC_T_AFFINE,
                   out_ectype=EC_T_AFFINE,
                   reduced=True,
                   ec2=True,
                   remove_last=True)[0].extend().as_str()
        vk_dict['vk_gamma_2'] = ECC.from_uint256(
                   self.pk['gamma_2'].reshape((-1,2,NWORDS_256BIT)),
                   in_ectype=EC_T_AFFINE,
                   out_ectype=EC_T_AFFINE,
                   reduced=True,
                   ec2=True,
                   remove_last=True)[0].extend().as_str()

        P = ECC.from_uint256(
                   self.pk['IC'],
                   in_ectype=EC_T_AFFINE,
                   out_ectype=EC_T_AFFINE,
                   reduced=True,
                   remove_last=True)
        vk_dict['IC'] = [x.extend().as_str() for x in P]
  
      return vk_dict

    def vars_to_vkbin(self):
      logging.error("Verifying Key  file %s can only be saved as .json\n", self.out_vk_f)
      sys.exit(1)
      vk_bin=None
      return vk_bin

    def timeStats(self, t):
      for s in t:
        for k, v in s.items():
           if k is not 'total':
             s[k] = str(round(v,2)) + '(' + str(round(100*v/s['total'],2)) + '%)'

    def logTimeResults(self):
      self.timeStats([self.t_S])

      logging.info('')
      logging.info('')
      logging.info('#################################### ')
      logging.info('Total Time to generate setup : %s seconds', round(self.t_S['total']))
      logging.info('')
      logging.info('- %s', self.t_S)
      logging.info('')
      logging.info('#################################### ')

if __name__ == "__main__":
    in_circuit_f = '../../data/prove-kyc.json'
    out_circuit_f = '../../data/prove-kyc.bin'
    #in_circuit_f = '../../data/circuit.json'
    #out_circuit_f = '../../data/circuit.bin'
    out_pk_f = '../../data/proving_key_prove-kyc.json'

    #GS = GrothSetup(in_circuit_f = in_circuit_f, out_circuit_f = out_circuit_f,
    GS = GrothSetup(in_circuit_f = out_circuit_f,
                    out_circuit_format=FMT_EXT, out_pk_f=out_pk_f, 
                    out_k_binformat=FMT_EXT, out_k_ecformat=EC_T_AFFINE, test_f=None)
    """
    if os.path.isfile(out_circuit_f):
       GS = GrothSetup(in_circuit_f=out_circuit_f, 
                       out_pk_f=out_pk_f, out_k_binformat=FMT_MONT,
                       out_k_ecformat=EC_T_AFFINE, test_f=None)
    else:
       GS = GrothSetup(in_circuit_f=in_circuit_f, out_circuit_f=out_circuit_f, 
                       out_circuit_format=FMT_MONT, out_pk_f=out_pk_f, out_k_binformat=FMT_MONT,
                       out_k_ecformat=EC_T_AFFINE, test_f=None)
    """

    GS.setup()

   

