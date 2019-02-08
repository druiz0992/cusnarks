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

NVCC = nvcc
GENCODE_ARCH_COMPUTE=-gencode arch=compute_60,code=sm_60
CFLAGS=-c -O2 -DLINUXINTEL64 -g -I $(INCLUDE_PATH) -w
CUFLAGS=--compiler-options '-fPIC -Wall -Wno-unused-but-set-variable -Wno-comment' -Xptxas -dlcm=cg --cudart shared $(GENCODE_ARCH_COMPUTE) 
CUBINFLAGS=-cubin -dlink $(GENCODE_ARCH_COMPUTE) 
LIBFLAGS= --shared -o 
LDFLAGS=-lm -lcudart

SRC = bigint.cu \
      bigint_device.cu

NVDISASM_FILE:=$(CUBIN_NAME:.cubin=.asm)

OBJ=$(SRC:.cu=.o)

clean:
	rm -f *.o *.cubin *.asm

build : so

cubin : 
	$(NVCC) $(CUBINFLAGS) $(SRC) -o $(CUBIN_NAME)
	nvdisasm $(CUBIN_NAME) > $(NVDISASM_FILE)

##################################################
so: $(OBJ)
	$(NVCC) $(LIBFLAGS) $(LIB_PATH)/$(CUSNARKS_LIB) $(OBJ) $(LDFLAGS)

##################################################
%.o : %.cu
	$(NVCC) $(CFLAGS) $(CUFLAGS) $*.cu -o $*.o	
