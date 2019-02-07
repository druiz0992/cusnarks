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
LIB_NAME = libcusnarks.so
CUBIN_NAME = cusnarks.cubin

dirs= $(CUSRC_PATH) \
      $(CTSRC_PATH) 

test_dirs = $(PYTST_PATH) 

SUBDIRS := $(dirs)
TEST_SUBDIRS := $(test_dirs)

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

MYMAKEFLAGS = 'CUSNARKS_PATH=$(CUSNARKS_PATH)'        \
              'INCLUDE_PATH=$(INCLUDE_PATH)'   \
              'OBJECT_PATH=$(OBJECT_PATH)'     \
              'LIB_PATH=$(LIB_PATH)'           \
              'CTSRC_PATH=$(CTSRC_PATH)'       \
              'PYSRC_PATH=$(PYSRC_PATH)'       \
              'PYTST_PATH=$(PYTST_PATH)'       \
              'CUSRC_PATH=$(CUSRC_PATH)'       \
              'LIB_NAME=$(LIB_NAME)'           \
              'CUBIN_NAME=$(CUBIN_NAME)'
              #'LIBS=$(LIBS)'                           \
              #'DEFINES=$(DEFINES)'                     \
              #'DEFINES_TEST=$(DEFINES_TEST)'                     \
              #'DEBUG_LEVEL=$(DEBUG_LEVEL)'             \
              #'EXTRA_FLAGS=$(EXTRA_FLAGS)'             \
              #'LDFLAGS=$(LDFLAGS)'                     \


#all:    
	#@for i in $(SUBDIRS); do \
	#echo "make all in $$i ..."; \
	#(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) all); done

#depend:
	#@for i in $(SUBDIRS) $(TEST_SUBDIRS); do \
	#echo "make depend in $$i..."; \
	#(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) depend); done

build:
	@for i in $(SUBDIRS); do \
	echo "make test in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) build); done

test:   
	@for i in $(TEST_SUBDIRS); do \
	echo "make test in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) test); done

clean:
	@for i in $(SUBDIRS) $(TEST_SUBDIRS); do \
	echo "clearing all in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) clean); done

cubin:
	cd $(CUSRC_PATH); $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) cubin

.PHONY:	test build clean
