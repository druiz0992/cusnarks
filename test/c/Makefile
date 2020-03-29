#
#
# ------------------------------------------------------------------
# Author     : David Ruiz
#
# File name  : Makefile
#
# Date       : 05/02/2019
#
# ------------------------------------------------------------------
#

INCLUDE_PATH = $(INCLUDE_PATH)
OBJECT_PATH = $(OBJECT_PATH)
LIB_PATH = $(LIB_PATH)
CTSRC_PATH = $(CTSRC_PATH)
PYSRC_PATH = $(PYSRC_PATH)
LD_FLAGS=$(LD_PATH)
CUSNARKS_LIB=$(CUSNARKS_LIB)
CUBIN_NAME=$(CUBIN_NAME)
AUX_INCLUDES=$(AUX_INCLUDES)
CTEST_PATH = $(CTEST_PATH)

LOG_LEVEL=$(LOG_LEVEL)

CC  = g++
CFLAGS=-c -g  -DLINUXINTEL64 -std=c++11 -I $(INCLUDE_PATH)  -w
EXTRA_CFLAGS2=-fopenmp -I $(AUX_INCLUDES) 
LINKFLAGS2=-L $(LIB_PATH)
LDFLAGS2=-lm -lcusnarks -lgomp

CSRC = test_utils_host.cpp

NVDISASM_FILE:=$(CUBIN_NAME:.cubin=.asm)
CUOBJ_FILE:=$(CUBIN_NAME:.cubin=.cuobj)

OBJ= $(CSRC:.cpp=.o)
TEST= $(subst .cpp,,$(CSRC))
SCRIPTS= $(subst .cpp,,$(CSRC2))

clean:
	rm -f *.o 

##################################################
test: $(OBJ)
	@for i in $(TEST); do \
	  ./$$i; \
	  echo ""; \
	done

##################################################
%.o : %.cpp
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS2) $*.cpp -o $*.o	
	$(CC) $(LINKFLAGS2) $*.o $(LDFLAGS2) -o $*
