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

CUSNARKS_PATH = ${CURDIR}
INCLUDE_PATH_TOP = ./src/main/include
INCLUDE_TEST_PATH_TOP = ./test/include
OBJECT_PATH_TOP = ./build
LIB_PATH_TOP = ./lib
CPSRC_PATH_TOP = ./src/main/c-wrappers
PYSRC_PATH_TOP = ./src/main/python

dirs= ./src/python

test_dirs = ./test/python 

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

#MYMAKEFLAGS = 'INCLUDE_PATH_TOP=$(INCLUDE_PATH_TOP)'   \
              #'INCLUDE_TEST_PATH_TOP=$(INCLUDE_TEST_PATH_TOP)' \
              #'CPSRC_PATH_TOP=$(CPSRC_PATH_TOP)'       \
              #'PYSRC_PATH_TOP=$(PYSRC_PATH_TOP)'       \
              #'OBJECT_PATH_TOP=$(OBJECT_PATH_TOP)'     \
              #'LIB_PATH_TOP=$(LIB_PATH_TOP)'           \
              #'LIBS=$(LIBS)'                           \
              #'DEFINES=$(DEFINES)'                     \
              #'DEFINES_TEST=$(DEFINES_TEST)'                     \
              #'DEBUG_LEVEL=$(DEBUG_LEVEL)'             \
              #'EXTRA_FLAGS=$(EXTRA_FLAGS)'             \
              #'LDFLAGS=$(LDFLAGS)'                     \
              #'CUSNARKS__PATH=$(CUSNARKS_PATH)'


#all:    
	#@for i in $(SUBDIRS); do \
	#echo "make all in $$i ..."; \
	#(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) all); done

#clean:
	#@for i in $(SUBDIRS) $(TEST_SUBDIRS); do \
	#echo "clearing all in $$i..."; \
	#(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) clean); done

#depend:
	#@for i in $(SUBDIRS) $(TEST_SUBDIRS); do \
	#echo "make depend in $$i..."; \
	#(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) depend); done

test:   
	@for i in $(TEST_SUBDIRS); do \
	echo "make test in $$i..."; \
	(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) test); done

.PHONY:	test
