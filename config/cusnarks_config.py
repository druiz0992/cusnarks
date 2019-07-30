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

CUSNARKS_SCRIPTS = os.getcwd()
CUSNARKS_HOME = CUSNARKS_SCRIPTS.replace('scripts/','')
CUSNARKS_REPO = CUSNARKS_HOME.replace('cusnarks/','')
CUSNARKS_ROOT = CUSNARKS_HOME + '/data')

snarkjs = { 
    'url'   :  'http://github.com/druiz0992/snarkjs.git',
    'folder' : CUSNARKS_HOME + 'iden3js'
}

rust-circomlib = {
    'url'   :  'http://github.com/adria0/rust-circom-experimental.git',
    'folder' :  CUSNARKS_HOME + 'rust-circom'
}

roots = {
  'nbits' : 20,
  'folder' : CUSNARKS_ROOTS + 'zpoly_roots_1M.bin'
}
 
def run():
    sys,stdout.write('#################################### ')
    sys.stdout.write('Configuring cusnarks....')
    sys.stdout.write('')
    sys.stdout.write('')

    download_repos()
    generate_roots()


def generate_roots():
    sys,stdout.write('#################################### ')
    sys.stdout.write('Generating roots of unity....')
    sys.stdout.write('Number of roots of unity have a dependency on the maximum number of constraints in your circuit. Default number roots of unity is 2^'+roots['nbits']+' roots of unity. Do you want a different number in the range of 2^20 and 2^28 ['+roots['nbits']+']?')
    sys.stdout.flush()
    b_root= sys.stdin.readline().rstrip()
    if b_root != str(roots['nbits'])  and b_repo == '':
       roots['nbits'] = int(b_root)

    sys.stdout.write('Default file to store roots is ' + roots['folder']. + ' Do you want to configure a new destination file [N/<file_name>]-N?')
    sys.stdout.flush()
    b_root = sys.stdin.readline().rstrip()

    if b_root != 'N' and b_root != 'n' and b_repo == '':
          roots['folder'] = b_repo
          if not roots['folder'].endswith('/'):
             roots['folder'] = roots['folder'] + '/'

    command = "./gen_roots " + roots['nbits'] + " " + roots['folder']
    os.system(command)
   
    
def download_repos(repo):
       sys.stdout.write('Default directory is ' + repo['folder']. + ' Do you want to configure a new destination folder [N/<folder_name>]-N?')
       sys.stdout.flush()
       b_repo = sys.stdin.readline().rstrip()

       if b_repo != 'N' and b_repo != 'n' and b_repo == '':
          repo['folder'] = b_repo
          if not repo['folder'].endswith('/'):
             repo['folder'] = repo['folder'] + '/'
       
       command = "git clone " + repo['url'] + " " + repo['folder']
       os.system(command)

def download_repos(): 
    sys,stdout.write('#################################### ')
    sys.stdout.write('Downloading required repos....')
    sys.stdout.write('snarkjs is required to verify proofs and to compute bindings during setup stage. You need to download druiz0992 fork. Do you want to download it now [Y/N] - Y?')
     sys.stdout.flush()
     b_download_repo = sys.stdin.readline().rstrip().capitalize()
    
     if b_download_repo == "Y" or b_download_repo == "":
        download_repo(snarkjs)

    sys.stdout.write('rust-circom-experimental is required to verify proofs and to compute bindings during setup stage. You need to download adria0 fork. Do you want to download it now [Y/N] - Y?')
     sys.stdout.flush()
     b_download_repo = sys.stdin.readline().rstrip().capitalize()
capitalize
     if b_download_repo == "Y" or b_download_repo == "":
        download_repo(snarkjs)

     sys.stdout.write('')
     sys.stdout.write('')
     

if __name__ == '__main__':
   run()


