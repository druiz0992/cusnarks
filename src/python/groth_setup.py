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
from multiproc  import *
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

ROOTS_1M_filename_npz = '../../data/zpoly_data_1M.npz'
ROOTS_1M_filename_bin = '../../data/zpoly_roots_1M.bin'

class GrothSetup(object):
    

    def __init__(self, curve='BN128', in_circuit_f=None, out_circuit_f=None, out_circuit_format=FMT_MONT,
                 out_pk_f=None, out_vk_f=None, out_k_binformat=FMT_MONT, out_k_ecformat=EC_T_AFFINE, test_f=None,
                 benchmark_f=None, seed=None, snarkjs=None, keep_f=None):
 
        # Check valid folder exists
        if keep_f is None:
            print ("Repo directory needs to be provided\n", file=self.log_f)
            sys.exit(1)

        self.keep_f = gen_reponame(keep_f, sufix="_SETUP") 

        logging.basicConfig(filename=self.keep_f + '/log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')

        if not use_pycusnarks :
          logging.error('PyCUSnarks shared library not found. Exiting...')
          sys.exit(1)

        if seed is not None:
          self.seed = seed
          random.seed(seed) 
        else:
          self.seed = random.randint(0,1<<32)

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

        self.alfabeta_f = self.keep_f + '/' + 'alfabeta.json'
        #self.test_f= self.keep_f + '/' +  test_f
        self.test_f= self.keep_f
        copy_input_files([in_circuit_f], self.keep_f)

        logging.info('#################################### ')
        logging.info('Staring Groth setupr with the follwing arguments :')
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
        logging.info('#################################### ')
         
        if self.in_circuit_f is not None:
           self.circuitRead()
           self.roots_rdc_u256_sh = RawArray(c_uint32,  (1 << self.pk['domainBits']) * NWORDS_256BIT)
           self.roots_rdc_u256 = np.frombuffer(
                    self.roots_rdc_u256_sh,
                    dtype=np.uint32).reshape((self.pk['domainBits'], NWORDS_256BIT))
           np.copyto(self.roots1M_rdc_u256, readU256DataFile_h(ROOTS_1M_filename_bin.encode("UTF-8"), 1<<20, 1<<self.pk['domainBits']) )

           #roots_rdc_u256 = field_roots_compute_h(bits)
        else:
           logging.error ("Required input circuit %s", self.log_f)
           sys.exit(1)

    def __del__(self):
       release_h()

    def circuitRead(self):
        # cir Json to u256
        if self.in_circuit_f.endswith('.json'):
           self.cir = cirjson_to_vars(self.in_circuit_f, self.in_circuit_format,
                                      self.out_circuit_format)

           #u256 to bin
           if self.out_circuit_f is not None:
              cir_u256 = cirvars_to_bin(self.cir)
              writeU256DataFile_h(cir_u256, self.out_circuit_f.encode("UTF-8"))
              del cir_u256

        elif self.in_circuit_f.endswith('.bin'):
             cir_u256 = readU256CircuitFile_h(self.in_circuit_f.encode("UTF-8"))
             self.cir = cirbin_to_vars(cir_u256)
             del cir_u256

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
        ZField.set_field(MOD_FIELD)
        
        #Init PK
        cirvars_to_pkvars(self.pk, self.cir)
        self.pk['k_binformat'] = self.out_k_binformat
        self.pk['k_ecformat'] =  self.out_k_ecformat


        prime = ZField.get_extended_p()

        self.toxic['t'] = ZFieldElExt(random.randint(1,prime.as_long()-1))

        self.calculatePoly()
        self.calculateEncryptedValuesAtT()

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
          snarkjs = self.launch_snarkjs("setup")


          worker = mp.Pool(processes=min(2,mp.cpu_count()-1))

          # Compare results
          r1 = worker.apply_async(pysnarks_compare, args=(test_pk_f, snarkjs['pk_f'], ['A', 'B1', 'B2', 'C', 'hExps', 'polsA', 'polsB',
                                                                   'polsC', 'vk_alfa_1', 'vk_beta_1', 'vk_beta_2',
                                                                   'vk_delta_1', 'vk_delta_2'], self.pk['nPublic'] ))
          r2 = worker.apply_async(pysnarks_compare, args=(test_vk_f, snarkjs['vk_f'], ['IC', 'vk_alfa_1', 'vk_alfabeta_12', 'vk_beta_2', 
                                                                   'vk_delta_2', 'vk_gamma_2'],0))

          pk_r = r1.get()
          vk_r = r2.get()

          worker.terminate()

          return pk_r and vk_r
        else:
          return True


    def calculatePoly(self):
        self.computeHeader()


        worker = mp.Pool(processes=min(3,mp.cpu_count()-1))

        r1 = worker.apply_async(cirr1cs_to_mpoly, args=(self.cir['R1CSA'], self.cir_header, self.cir['cirformat'], 1))

        r2 = worker.apply_async(cirr1cs_to_mpoly, args=(self.cir['R1CSB'], self.cir_header,self.cir['cirformat'], 0))

        r3 = worker.apply_async(cirr1cs_to_mpoly, args=(self.cir['R1CSC'], self.cir_header, self.cir['cirformat'],0))

        self.pk['polsA'] = r1.get()
        self.pk['polsA_nWords'] = self.pk['polsA'].shape[0]
        self.cir['R1CSA'] = None
        self.pk['polsB'] = r2.get()
        self.pk['polsB_nWords'] = self.pk['polsB'].shape[0]
        self.cir['R1CSB'] = None
        self.pk['polsC'] = r3.get()
        self.pk['polsC_nWords'] = self.pk['polsC'].shape[0]
        self.cir['R1CSC'] = None

        worker.terminate()

        return

        """
        self.pk['polsA']  = self.r1cs_to_mpoly(self.cir['R1CSA'], 1)
        self.pk['polsA_nWords'] = self.pk['polsA'].shape[0]
        self.cir['R1CSA'] = None
        self.pk['polsB'] = self.r1cs_to_mpoly(self.cir['R1CSB'], 0)
        self.pk['polsB_nWords'] = self.pk['polsB'].shape[0]
        self.cir['R1CSB'] = None
        self.pk['polsC'] = self.r1cs_to_mpoly(self.cir['R1CSC'], 0)
        self.pk['polsC_nWords'] = self.pk['polsC'].shape[0]
        self.cir['R1CSC'] = None
        """


    def r1cs_to_mpoly(self, r1cs, fmat, extend):
        to_mont = 0
        ZField.set_field(MOD_FIELD)
        pidx = ZField.get_field()
        if fmat == ZUtils.FEXT:
           to_mont = 1

        poly_len = r1cs_to_mpoly_len_h(r1cs, self.cir_header, extend)
        pols = r1cs_to_mpoly_h(poly_len, r1cs, self.cir_header, to_mont, pidx, extend)
        
        return pols

    def evalLagrangePoly(self, bits):
       """
        m : int
        t : ZFieldElRedc
       """
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

       return z.as_uint256(), u_u256
   
    def calculateEncryptedValuesAtT(self):
      a_t_u256, b_t_u256, c_t_u256, z_t_u256 = self.calculateValuesAtT()

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
      self.pk['alfa_1'] = G1 * self.toxic['kalfa']
      self.pk['beta_1'] = G1 * self.toxic['kbeta']
      self.pk['delta_1'] = G1 * self.toxic['kdelta']

      self.pk['alfa_1'] = ec_jac2aff_h(G1.as_uint256(self.pk['alfa_1']).reshape(-1),ZField.get_field())
      self.pk['beta_1'] = ec_jac2aff_h(G1.as_uint256(self.pk['beta_1']).reshape(-1),ZField.get_field())
      self.pk['delta_1'] = ec_jac2aff_h(G1.as_uint256(self.pk['delta_1']).reshape(-1),ZField.get_field())
    
      self.pk['beta_2'] = G2 * self.toxic['kbeta']
      self.pk['delta_2'] = G2 * self.toxic['kdelta']
      self.pk['gamma_2'] = G2 * self.toxic['kgamma']
    
      self.pk['beta_2'] = ec2_jac2aff_h(G2.as_uint256(self.pk['beta_2']).reshape(-1),ZField.get_field())
      self.pk['delta_2'] = ec2_jac2aff_h(G2.as_uint256(self.pk['delta_2']).reshape(-1),ZField.get_field())
      self.pk['gamma_2'] = ec2_jac2aff_h(G2.as_uint256(self.pk['gamma_2']).reshape(-1),ZField.get_field())

      self.ecbn128    =  ECBN128(self.pk['domainSize']+3,seed=self.seed)
      self.ec2bn128    = EC2BN128(self.pk['nVars']+1,seed=self.seed)
   
      # a_t, b_t and c_t are in ext
      sorted_idx = sortu256_idx_h(a_t_u256)
      ecbn128_samples = np.concatenate((a_t_u256[sorted_idx],G1.as_uint256(G1)[:2]))
      #ecbn128_samples = np.concatenate((a_t_u256,G1.as_uint256(G1)[:2]))
      self.pk['A'],_ = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.pk['A'] = ec_jac2aff_h(self.pk['A'].reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['A'] = np.reshape(self.pk['A'],(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.pk['A'] = np.reshape(self.pk['A'],(-1,NWORDS_256BIT))
      self.pk['A_nWords'] = np.uint32(self.pk['A'].shape[0] * NWORDS_256BIT*2/3)

      sorted_idx = sortu256_idx_h(b_t_u256)
      ecbn128_samples[:-2] = b_t_u256[sorted_idx]
      self.pk['B1'],t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.pk['B1'] = ec_jac2aff_h(self.pk['B1'].reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['B1'] = np.reshape(self.pk['B1'],(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.pk['B1']=np.reshape(self.pk['B1'],(-1,NWORDS_256BIT))
      self.pk['B1_nWords'] = np.uint32(self.pk['B1'].shape[0] * NWORDS_256BIT*2/3)

      ec2bn128_samples = np.concatenate((b_t_u256[sorted_idx],G2.as_uint256(G2)[:4]))
      #ec2bn128_samples = np.concatenate((b_t_u256,G2.as_uint256(G2)[:4]))
      self.pk['B2'],t = ec_sc1mul_cuda(self.ec2bn128, ec2bn128_samples, ZField.get_field(), ec2=True)
      self.pk['B2'] = ec2_jac2aff_h(self.pk['B2'].reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['B2'] = np.reshape(self.pk['B2'],(-1,6,NWORDS_256BIT))[unsorted_idx]
      #self.B2 = np.reshape(self.B2,(-1,6,NWORDS_256BIT))
      self.pk['B2'] = np.reshape(self.pk['B2'],(-1,NWORDS_256BIT))
      #ECC.from_uint256(self.B2.reshape((-1,2,8))[0:3],reduced=True, in_ectype=2, out_ectype=2,ec2=True)[0].extend().as_list()
      self.pk['B2_nWords'] = np.uint32(self.pk['B2'].shape[0] * NWORDS_256BIT*4/6)

      ZField.set_field(MOD_FIELD)
      pidx = ZField.get_field()
      ps_u256 = GrothSetupComputePS_h(self.toxic['kalfa'].reduce().as_uint256(), self.toxic['kbeta'].reduce().as_uint256(),
                                      toxic_invDelta.reduce().as_uint256(),
                                      a_t_u256, b_t_u256, c_t_u256, self.pk['nPublic']+1, self.pk['nVars'], pidx )
      ZField.set_field(MOD_GROUP)
      sorted_idx = sortu256_idx_h(ps_u256)
      ecbn128_samples = np.concatenate((ps_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.pk['C'],t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.pk['C'] = ec_jac2aff_h(self.pk['C'].reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['C'] = np.reshape(self.pk['C'],(-1,3,NWORDS_256BIT))[unsorted_idx]
      #ECC.from_uint256(self.C[12],reduced=True, in_ectype=2, out_ectype=2)[0].extend().as_list()
      self.pk['C']=np.concatenate((np.zeros(((self.pk['nPublic']+1)*3,NWORDS_256BIT),dtype=np.uint32),np.reshape(self.pk['C'],(-1,NWORDS_256BIT))))
      self.pk['C_nWords'] = np.uint32(self.pk['C'].shape[0] * NWORDS_256BIT*2/3)


      maxH = self.pk['domainSize']+1
      self.pk['hExps'] = np.zeros((maxH,NWORDS_256BIT),dtype=np.uint32)

      ZField.set_field(MOD_FIELD)
      pidx = ZField.get_field()
      zod_u256 = montmult_h(toxic_invDelta.reduce().as_uint256(), z_t_u256, pidx)
      eT_u256 = GrothSetupComputeeT_h(self.toxic['t'].reduce().as_uint256(), zod_u256, maxH, pidx)

      ZField.set_field(MOD_GROUP)
      sorted_idx = sortu256_idx_h(eT_u256)
      ecbn128_samples = np.concatenate((eT_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.pk['hExps'],t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.pk['hExps'] = ec_jac2aff_h(self.pk['hExps'].reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['hExps'] = np.reshape(self.pk['hExps'],(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.pk['hExps']=np.reshape(self.pk['hExps'],(-1,NWORDS_256BIT))
      self.pk['hExps_nWords'] = np.uint32(self.pk['hExps'].shape[0] * NWORDS_256BIT*2/3)


      ZField.set_field(MOD_FIELD)
      pidx = ZField.get_field()
      ps_u256 = GrothSetupComputePS_h(self.toxic['kalfa'].reduce().as_uint256(), self.toxic['kbeta'].reduce().as_uint256(),
                                      toxic_invGamma.reduce().as_uint256(),
                                      a_t_u256, b_t_u256, c_t_u256, 0, self.pk['nPublic']+1, pidx )
      ZField.set_field(MOD_GROUP)
      sorted_idx = sortu256_idx_h(ps_u256)
      ecbn128_samples = np.concatenate((ps_u256[sorted_idx], G1.as_uint256(G1)[:2]))
      self.pk['IC'],t = ec_sc1mul_cuda(self.ecbn128, ecbn128_samples, ZField.get_field())
      self.pk['IC'] = ec_jac2aff_h(self.pk['IC'].reshape(-1),ZField.get_field())
      unsorted_idx = np.argsort(sorted_idx)
      self.pk['IC'] = np.reshape(self.pk['IC'],(-1,3,NWORDS_256BIT))[unsorted_idx]
      self.pk['IC'] = np.uint32(np.reshape(self.pk['IC'],(-1,NWORDS_256BIT)))

    def calculateValuesAtT(self):
       # Required z_t es Ext, polsA/B/C are in Mont format
       # u is Mont
       z_t_u256, u = self.evalLagrangePoly(self.pk['domainBits'])

       pidx = ZField.get_field()

       worker = mp.Pool(processes=min(3,mp.cpu_count()-1))

       r1 = worker.apply_async(mpoly_madd_h, args = (self.pk['polsA'], u.reshape(-1), self.pk['nVars'], pidx))
       r2 = worker.apply_async(mpoly_madd_h, args =(self.pk['polsB'], u.reshape(-1), self.pk['nVars'], pidx))
       r3 = worker.apply_async(mpoly_madd_h, args = (self.pk['polsC'], u.reshape(-1), self.pk['nVars'], pidx))

       a_t_u256 = r1.get()
       b_t_u256 = r2.get()
       c_t_u256 = r3.get()

       worker.terminate()
      
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



    def write_pk(self):

       if self.out_pk_f.endswith('.json') :
         pk_dict = pkvars_to_json(self.out_k_binformat, self.out_k_ecformat,self.pk)
         pk_json = json.dumps(pk_dict, indent=4, sort_keys=True)
         #if os.path.exists(self.out_pk_f):
         #    os.remove(self.out_pk_f)
         f = open(self.out_pk_f, 'w')
         print(pk_json, file=f)
         f.close()

       elif self.out_pk_f.endswith('bin') :
         pk_bin = pkvars_to_bin(self.out_k_binformat, self.out_k_ecformat, self.pk)
         writeU256DataFile_h(pk_bin, self.out_pk_f.encode("UTF-8"))

       else :
         logging.info ("No valid proving key file  %s provided", self.out_pk_f)
         os.exit(1)
        
    def write_vk(self):
       #Fill alfa and beta values and call snarkjs to compute vk value
       vk_dict = self.vars_to_vkdict(alfabeta=True)
       vk_json = json.dumps(vk_dict, indent=4, sort_keys=True)
       #if os.path.exists(self.alfabeta_f):
       #     os.remove(self.alfabeta_f)
       f = open(self.alfabeta_f, 'w')
       print(vk_json, file=f)
       f.close()

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

    def vars_to_vkdict(self, alfabeta=False):
      # TODO : only suported formats for vk are .json, affine and extended 
      vk_dict = {}
      ZField.set_field(MOD_GROUP)
      vk_dict['vk_alfa_1'] = ECC.from_uint256(self.pk['alfa_1'], in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)[0].extend().as_str()
      vk_dict['vk_beta_2'] = ECC.from_uint256(self.pk['beta_2'].reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True, ec2=True)[0].extend().as_str()

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
  
        vk_dict['vk_delta_2'] = ECC.from_uint256(self.pk['delta_2'].reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True, ec2=True)[0].extend().as_str()
        vk_dict['vk_gamma_2'] = ECC.from_uint256(self.pk['gamma_2'].reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True, ec2=True)[0].extend().as_str()

        P = ECC.from_uint256(self.pk['IC'], in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)
        vk_dict['IC'] = [x.extend().as_str() for x in P]
  
      return vk_dict

    def vars_to_vkbin(self):
      logging.error("Verifying Key  file %s can only be saved as .json\n", self.out_vk_f)
      self.log_f.flush()
      sys.exit(1)
      vk_bin=None
      return vk_bin


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

   

