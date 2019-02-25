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

LOG_LEVEL=$(LOG_LEVEL)

NVCC = nvcc
CC  = c++
GENCODE_ARCH_COMPUTE=-gencode arch=compute_60,code=sm_60
CFLAGS=-c -O2 -DLINUXINTEL64 -g -I $(INCLUDE_PATH) -w
EXTRA_CFLAGS=-I $(AUX_INCLUDES)
CUFLAGS= -DLOG_LEVEL=$(LOG_LEVEL) -std=c++11 -rdc=true --compiler-options '-fPIC -Wall -Wno-unused-but-set-variable -Wno-comment' -Xptxas -v --cudart shared $(GENCODE_ARCH_COMPUTE) 
CUBINFLAGS=-std=c++11 -cubin -dlink $(GENCODE_ARCH_COMPUTE) 
LINKFLAGS= -arch=sm_60 --shared -o 
LDFLAGS=-lm -lcudart
NVDISASM = nvdisasm
NVDISASM_FLAGS = --no-dataflow --print-code --print-line-info --separate-functions
CUOBJDUMP = cuobjdump
CUOBJDUMP_FLAGS = -ptx -sass 

CUSRC = cusnarks_kernel.cu \
        u256.cu \
        u256_device.cu \
        ecbn128.cu \
        ecbn128_device.cu \
        ecbn128_2.cu \
        ecbn128_2_device.cu \
        log.cu

CSRC = rng.cpp 

NVDISASM_FILE:=$(CUBIN_NAME:.cubin=.asm)
CUOBJ_FILE:=$(CUBIN_NAME:.cubin=.cuobj)

OBJ=$(CUSRC:.cu=.o) \
    $(CSRC:.cpp=.o)

clean:
	rm -f *.o *.cubin *.asm *.cuobj

build : so

cubin : 
	$(NVCC) $(CUBINFLAGS) $(EXTRA_CFLAGS) $(CUSRC) -o $(CUBIN_NAME)
	$(NVDISASM) $(CUBIN_NAME) $(NVDISASM_FLAGS) > $(NVDISASM_FILE)
	$(CUOBJDUMP) $(CUBIN_NAME) $(CUOBJDUMP_FLAGS) > $(CUOBJ_FILE)

##################################################
so: $(OBJ)
	$(NVCC) $(LINKFLAGS) $(LIB_PATH)/$(CUSNARKS_LIB) $(OBJ) $(LDFLAGS)

##################################################
%.o : %.cu
	$(NVCC) $(CFLAGS) $(EXTRA_CFLAGS) $(CUFLAGS) $*.cu -o $*.o	

##################################################
%.o : %.cpp
	$(NVCC) $(CFLAGS) $(EXTRA_CFLAGS) $(CUFLAGS) $*.cpp -o $*.o	
