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
// File name  : cusnarks_config.py
//
// Date       : 30/07/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Configure cusnarks 
//
// ------------------------------------------------------------------

"""
import argparse
import os
import sys
import re

CUSNARKS_CONFIG = os.path.dirname(os.path.realpath(__file__))+'/'
CUSNARKS_HOME = CUSNARKS_CONFIG.replace('config/','')
CUSNARKS_REPO = CUSNARKS_HOME.replace('cusnarks/','')
CUSNARKS_ROOTS = CUSNARKS_HOME + 'data/'
CUSNARKS_CIRCUITS = CUSNARKS_HOME + 'circuits/'

CUSNARKS_ROOTS_F = CUSNARKS_CONFIG + '.root_f'
CUSNARKS_ROOTS_N = CUSNARKS_CONFIG + '.nroots'

snarkjs = { 
    'url'   :  'http://github.com/druiz0992/snarkjs.git',
    'folder' : CUSNARKS_REPO + 'third_party_libs/snarkjs',
    'Q1'     : 'Snarkjs is required to verify proofs and to compute bindings during Setup stage.\n' + \
                'Cusnarks uses druiz0992\'s fork. Do you want to download it now [Y/N] - Y?\n',
    'Q2'     :  '\n\nHave you downloaded snarkjs already?\n' + \
                'If so, please enter the directory location. [N/<directory location>] - N?\n',
    'Q3'     :  '/cli.js doesn\'t exist. You can download druiz0992/snarkjs manually later on\n',
    'file'   : 'cli.js'
}

rust_circom = {
    'url'   :  'http://github.com/adria0/rust-circom-experimental.git',
    'folder' :  CUSNARKS_REPO + 'third_party_libs/rust-circom-experimental',
    'Q1'     : '\n\n\nrust-circom-experimental is used to accelerate the compilation of snarks circuits.\n'+ \
               '\n\nCusnarks uses adria0\'s fork. Do you want to download it now [Y/N] - Y?\n',
    'Q2'     : 'Have you downloaded rust_circom already?\n' + \
               'If so, please enter the directory location. [N/<directory location>] - N?\n',
    'Q3'     :   'doesn\'t exist. You can download adria0/rust-circom manually later on\n',
    'file'   :  'interop/circuits/cuda/../../../target/release/circom2'
 
}

roots = {
  'nbits' : 20,
  'folder' : CUSNARKS_ROOTS + 'zpoly_roots_1M.bin'
}

config = {
   'folder' : CUSNARKS_CONFIG + 'cusnarks.conf'
}

circuits = {
  'folder' : CUSNARKS_CIRCUITS
}

def run():
    sys.stdout.write('####################################\n')
    sys.stdout.write('Configuring cusnarks....\n\n')

    #download_repos([snarkjs, rust_circom])
    generate_roots()
    generate_circuit_folder()
    generate_config_f()

def generate_circuit_folder():
    sys.stdout.write('####################################\n')
    sys.stdout.write('Generating circuit folder....\n\n')
    sys.stdout.write('Type directory location where you want to save your circuits.\n')
    sys.stdout.write('Default location is '+ circuits['folder'] + '. Do you want a different location?\n')
    sys.stdout.write('If you provide a different directory name, it will be created if it doesn\'t exist [N/<type folder location>]-N?\n')
    sys.stdout.flush()
    b_folder= sys.stdin.readline().rstrip()
    # If not default, retrieve new circuit folder name
    if b_folder != 'N' and b_folder != 'n' and b_folder != '':
      if not b_folder.endswith('/'):
         b_folder = b_folder + '/'
      circuits['folder'] = b_folder
      if not os.path.exists(b_folder):
        os.makedirs(b_folder)
    elif b_folder == "":
        sys.stdout.write(b_folder+'\n')

def generate_config_f():
    sys.stdout.write('####################################\n')
    sys.stdout.write('Generating config file....\n\n')
    f = open(config['folder'],'w')
    print("###### FILE GENERATED AUTOMATICALLY ####\n\n\n", file=f)
    print("CIRCUITS : " + circuits['folder'] +'\n\n',file=f)
    print("SNARKJS : " +snarkjs['folder'] + '/' + snarkjs['file']+'\n\n',file=f)
    print("RUST-CIRCOM : " + rust_circom['folder'] + '\n\n', file=f)
    print("ROOTS_FILE : " + roots['folder'] + '\n\n', file=f)
    print("N_ROOTS : " + str(roots['nbits']) + '\n\n', file=f)
    f.close()

    sys.stdout.write('####################################\n\n')

    f = open(CUSNARKS_ROOTS_F,'w')
    print(roots['folder'],file=f)
    f.close()

    f = open(CUSNARKS_ROOTS_N,'w')
    print(roots['nbits'],file=f)
    f.close()

def get_roots_file():
    return parse_configfile(config['folder'],'ROOTS_FILE')

def get_circuits_folder():
    return parse_configfile(config['folder'],'CIRCUITS')

def get_snarkjs_folder():
    return parse_configfile(config['folder'],'SNARKJS')

def get_rust_folder():
    return parse_configfile(config['folder'],'RUST-CIRCOM')

def get_n_roots():
    return int(parse_configfile(config['folder'],'N_ROOTS'))

def parse_configfile(fname, label):
    f = open(fname,'r')
    lines = f.readlines()
    f.close()
  
    pattern='^'+label+' : ([\w|/|\.|-]+)'
    for l in lines:
      match = re.search(pattern, l.rstrip())
      if match:
        return match.group(1)
   

def generate_roots():
    sys.stdout.write('####################################\n\n')
    sys.stdout.write('Generating roots of unity....\n\n')
    sys.stdout.write('Number of roots of unity imposes a limit on the maximum number of constraints\n')
    sys.stdout.write('in your circuit. Default number roots of unity is 2^'+str(roots['nbits'])+'.\n')
    sys.stdout.write('Do you want a different number in the range of 2^20 and 2^28 ['+str(roots['nbits'])+']?\n')
    sys.stdout.flush()
    b_root= sys.stdin.readline().rstrip()
    if b_root != '':
        while True:
          try :
             # Try until input is valid
             if int(b_root) > 28 or int(b_root) < 20 :
                sys.stdout.write('Do you want a different number in the range of 2^20 and 2^28 ['+str(roots['nbits'])+']?\n')
                b_root= sys.stdin.readline().rstrip()
             else:
                break
          except :
                sys.stdout.write('Do you want a different number in the range of 2^20 and 2^28 ['+str(roots['nbits'])+']?\n')
                b_root= sys.stdin.readline().rstrip()
    else:
      b_root = roots['nbits']

    roots['nbits'] = int(b_root)

    sys.stdout.write('Default file to store roots is ' + roots['folder'] + '.\n')
    sys.stdout.write('Do you want to configure a new destination file [N/<file_name>]-N?\n')
    sys.stdout.flush()
    b_root = sys.stdin.readline().rstrip()

    # if not default filename
    if b_root != 'N' and b_root != 'n' and b_root != '':
          fname = b_root.split('/')[-1]
          folder = b_root.replace(fname,'')
          if not os.path.exists(folder):
            os.makedirs(folder)
          roots['folder'] = b_root

    sys.stdout.write('Generating 2^'+str(roots['nbits']) + ' roots of unity in ' +roots['folder']+'\n')
    command = "./gen_roots " + str(roots['nbits']) + " " + roots['folder']
    os.system(command)
    sys.stdout.write('####################################\n\n')
   
    
def download_repo(repo):
       sys.stdout.write('Default directory is ' + repo['folder']+'\n')
       sys.stdout.write('Do you want to configure a new destination folder [N/<folder_name>]-N?\n')
       sys.stdout.flush()
       b_repo = sys.stdin.readline().rstrip()

       if b_repo != 'N' and b_repo != 'n' and b_repo != '':
          repo['folder'] = b_repo
          if not repo['folder'].endswith('/'):
             repo['folder'] = repo['folder'] + '/'
       elif b_repo =="":
          sys.stdout.write(repo['folder']+'\n')

       command = "git clone " + repo['url'] + " " + repo['folder']
       os.system(command)
       sys.stdout.write("Read "+repo['url'] + ' README.md to install it.\n')

def download_repos(repos): 
    sys.stdout.write('####################################\n\n')
    sys.stdout.write('Downloading required repos....\n\n')
    for r in repos:
       sys.stdout.write(r['Q1'])
       sys.stdout.flush()
       b_download_repo = sys.stdin.readline().rstrip()
    
       if b_download_repo == "Y" or b_download_repo =="y" or b_download_repo == "":
          if b_download_repo == "":
            sys.stdout.write("Y\n")
          download_repo(r)
       else:
         sys.stdout.write(r['Q2'])
         sys.stdout.flush()
         b_repo = sys.stdin.readline().rstrip()
         if b_repo != "N" and b_repo =="n" and b_download_repo == "":
            r['folder'] = b_repo
            if not r['folder'].endswith('/'):
             r['folder'] = r['folder']+'/'
            if not os.path.exists(r['folder']+'/' + r['file']):
             sys.stdout.write(r['folder'] + r['Q3'])
           
    sys.stdout.write('\n\n####################################\n\n')
     

if __name__ == '__main__':
   run()
