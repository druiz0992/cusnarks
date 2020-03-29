# ------------------------------------------------------------------
#
#    Copyright 2018 0kims association.
#
#    This file is part of cusnarks.
#
#    cusnarks is a free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published by the
#    Free Software Foundation, either version 3 of the License, or (at your option)
#    any later version.
#
#    cusnarks is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
#    more details.
#
#    You should have received a copy of the GNU General Public License along with
#    cusnarks. If not, see <https://www.gnu.org/licenses/>.
#
#
# ------------------------------------------------------------------
# Author     : David Ruiz
#
# File name  : Makefile
#
# Date       : 31/01/2019
#
# ------------------------------------------------------------------
#
###########  MakeFile.env  ##########
# Top level pattern, include by Makefile of child directory
# in which variable like TOPDIR, TARGET or LIB may be needed

#CC=gcc
MAKE=make
NPM=npm i
CARGO=cargo build --release
SHELL=/bin/sh

#####
# Define Paths

CUSNARKS_PATH = ${PWD}
INCLUDE_PATH = $(CUSNARKS_PATH)/src/cuda
OBJECT_PATH = $(CUSNARKS_PATH)/build
LIB_PATH = $(CUSNARKS_PATH)/lib
CTSRC_PATH = $(CUSNARKS_PATH)/src/cython
PYSRC_PATH = $(CUSNARKS_PATH)/src/python
PYTST_PATH = $(CUSNARKS_PATH)/test/python
CUSRC_PATH = $(CUSNARKS_PATH)/src/cuda
CTEST_PATH = $(CUSNARKS_PATH)/test/c
CTEST_IDEAS_PATH = $(CUSNARKS_PATH)/test/ideas
CONFIG_PATH= $(CUSNARKS_PATH)/config

AUX_PATH = $(CUSNARKS_PATH)/third_party_libs

PCG_PATH = $(AUX_PATH)/pcg-cpp/test-high
PCG_REPO = https://github.com/imneme/pcg-cpp.git
PCG_INCLUDE = $(AUX_PATH)/pcg-cpp/include

SNARKJS_PATH = $(AUX_PATH)/snarkjs
SNARKJS_REPO = https://github.com/druiz0992/snarkjs.git

RUST_CIRCOM_PATH = $(AUX_PATH)/za
RUST_CIRCOM_REPO = http://github.com/iden3/za.git

CUSNARKS_LIB = libcusnarks.so
CUBIN_NAME = cusnarks.cubin

dirs= $(CUSRC_PATH) \
      $(CTSRC_PATH) 

aux_cdirs  = $(PCG_PATH) 
aux_jsdirs = $(SNARKJS_PATH)
aux_rdirs  = $(RUST_CIRCOM_PATH)

test_dirs = $(CTEST_PATH) \
          $(PYTST_PATH) 
            

aux_repos = $(PCG_REPO) \
            $(SNARKJS_REPO) \
            $(RUST_CIRCOM_REPO)

config_dirs = $(CONFIG_PATH)

AUX_INCLUDES = $(PCG_INCLUDE)

SUBDIRS := $(dirs)
TEST_SUBDIRS := $(test_dirs)
CONFIG_SUBDIRS := $(config_dirs)
AUX_CSUBDIRS := $(aux_cdirs)
AUX_JSSUBDIRS := $(aux_jsdirs)
AUX_RSUBDIRS := $(aux_rdirs)
AUX_REPOS := $(aux_repos)

#####
# Define Options
#LIBS = -lm 
#DEFINES =
#DEFINES_TEST = 
#DEFINES = DEBUG_PYTHON CHECK_ERRORS UNIT_TEST T_DOUBLE=1 LOG_LEVEL=255
#DEBUG_LEVEL = -g
#EXTRA_FLAGS = -Wall -fPIC
#CFLAGS = $(DEBUG_LEVEL) $(EXTRA_FLAGS) $(INCLUDE_PATH) $(DEFINES)
#LDFLAGS = -shared

# LOG LEVEL : 4 -> No Log
#             3 -> Error
#             2 -> Warning
#             1 -> Info
#             0 -> Debug
#

#DEFINES
# -DPARALLEL_EN, -DCU_ASM, -DCU_ASM, -DCU_ASM_ECADD
DEFINES = -DPARALLEL_EN  -DCU_ASM 
DEFINES_DEBUG = -DCU_ASM

MYMAKEFLAGS = 'CUSNARKS_PATH=$(CUSNARKS_PATH)'        \
              'INCLUDE_PATH=$(INCLUDE_PATH)'   \
              'OBJECT_PATH=$(OBJECT_PATH)'     \
              'LIB_PATH=$(LIB_PATH)'           \
              'CTSRC_PATH=$(CTSRC_PATH)'       \
              'PYSRC_PATH=$(PYSRC_PATH)'       \
              'PYTST_PATH=$(PYTST_PATH)'       \
              'CUSRC_PATH=$(CUSRC_PATH)'       \
              'CTEST_PATH=$(CTEST_PATH)'       \
              'CUSNARKS_LIB=$(CUSNARKS_LIB)'           \
              'CUBIN_NAME=$(CUBIN_NAME)'      \
              'AUX_INCLUDES=$(AUX_INCLUDES)'  \
              'DEFINES=$(DEFINES)'   \
              'DEFINES_DEBUG=$(DEFINES_DEBUG)'  
  
build:
	@for i in $(SUBDIRS); do \
		echo "make build in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) build); done


all: third_party_libs build

force_all: clean third_party_libs_clean all

test:   
	@for i in $(TEST_SUBDIRS); do \
	echo "make test in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) test); done

test_cpu:   
	@for i in $(CTEST_PATH); do \
	echo "make test in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) test); done

test_gpu:   
	@for i in $(PYTST_PATH); do \
	echo "make test in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) test); done

debug_gpu:   
	@for i in $(CTEST_IDEAS_PATH); do \
	echo "make test in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) test); done
config:   
	@for i in $(CONFIG_SUBDIRS); do \
	echo "make test in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) config); done

clean:
	@for i in $(SUBDIRS) $(TEST_SUBDIRS) $(SCRIPTS_SUBDIRS) $(CTEST_IDEAS_PATH); do \
	echo "clearing all in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) clean); done

cubin:
	cd $(CUSRC_PATH); $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) cubin

third_party_libs:
	echo "checking third pary libs...";
	if ! test -d $(AUX_PATH); \
		then mkdir $(AUX_PATH); cd $(AUX_PATH); for j in $(AUX_REPOS); do git clone $$j; done;  fi
	@for i in $(AUX_CSUBDIRS); do \
		(cd $$i; $(MAKE)); done
	@for i in $(AUX_JSSUBDIRS); do \
		(cd $$i; $(NPM)); done
	@for i in $(AUX_RSUBDIRS); do \
		(cd $$i; $(CARGO)); done

clib:
	echo "checking third pary libs...";
	if ! test -d $(AUX_PATH); \
		then mkdir $(AUX_PATH); cd $(AUX_PATH); for j in $(PCG_REPO); do git clone $$j; done;  fi
	@for i in $(AUX_CSUBDIRS); do \
		(cd $$i; $(MAKE)); done
	if ! test -d $(LIB_PATH); \
		then mkdir $(LIB_PATH); fi
	@for i in $(CUSRC_PATH); do \
		echo "make build in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) build); done

	

third_party_libs_clean:
	if test -d $(AUX_PATH); then rm -rf $(AUX_PATH); fi

.PHONY:	config test build clean all force_all third_party_libs_clean third_party_libs test_cpu test_gpu clib debug_gpu

