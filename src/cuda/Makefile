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
DEFINES=$(DEFINES)
DEFINES_DEBUG=$(DEFINES_DEBUG)

LOG_LEVEL=$(LOG_LEVEL)

NVCC = nvcc
CC  = g++
ASM = nasm
GENCODE_ARCH_COMPUTE=-gencode arch=compute_61,code=sm_61
DEBUG=0
ADX_SUPPORT=$(shell cat /proc/cpuinfo | grep -m1 -oP 'adx')
ifeq ($(DEBUG),1)
  CFLAGS=-c -g -DLINUXINTEL64  -DTEST_MODE $(DEFINES_DEBUG) -std=c++11 -I $(INCLUDE_PATH) -w
  CUFLAGS=-rdc=true -G -O3 --source-in-ptx -keep --compiler-options '-fPIC -Wall -Wno-unused-but-set-variable -Wno-comment' -Xptxas -v -Xptxas -maxrregcount=120 --cudart shared $(GENCODE_ARCH_COMPUTE)
else
  CFLAGS=-c -O3 -DLINUXINTEL64 -lineinfo $(DEFINES) -std=c++11 -I $(INCLUDE_PATH) -w
  CUFLAGS=-rdc=true --compiler-options '-fPIC -Wall -Wno-unused-but-set-variable -Wno-comment' -Xptxas -v -Xptxas -maxrregcount=120 --cudart shared $(GENCODE_ARCH_COMPUTE)
endif

ASMFLAGS=-f elf64
EXTRA_CFLAGS=-I $(AUX_INCLUDES)
EXTRA_CFLAGS2=-march=native -fopenmp -I $(AUX_INCLUDES) -shared -fPIC
CUBINFLAGS=-std=c++11 -cubin -dlink $(GENCODE_ARCH_COMPUTE) 
LINKFLAGS= -arch=sm_61 --shared -o 
LINKFLAGS2=--shared -o 
LDFLAGS=-lm -lcudart -lgomp
LDFLAGS2=-lm -lgomp
NVDISASM = nvdisasm
NVDISASM_FLAGS = --no-dataflow --print-code --print-line-info --separate-functions
CUOBJDUMP = cuobjdump
CUOBJDUMP_FLAGS = -ptx -sass 

CUSRC = cusnarks_kernel.cu \
        u256.cu \
        u256_device.cu \
        ecbn128.cu \
        ecbn128_device.cu \
	ec2bn128.cu \
	zpoly.cu \
	zpoly_device.cu \
        z1_device.cu \
        z2_device.cu \
        async_buffer.cu

CSRC = rng.cpp \
       utils_host.cpp \
       constants.cpp \
       log.cpp

ifeq ($(ADX_SUPPORT),adx)
CASMSRC = fr_adx.casm \
          fq_adx.casm
else
CASMSRC = fr.casm \
          fq.casm
endif

#Check if CUDA tools are installed.
CUDA_INSTALLED = $(shell which $(NVCC))


NVDISASM_FILE:=$(CUBIN_NAME:.cubin=.asm)
CUOBJ_FILE:=$(CUBIN_NAME:.cubin=.cuobj)


ifeq ($(CUDA_INSTALLED),)
OBJ= $(CSRC:.cpp=.o) \
     $(CASMSRC:.casm=.o)
else
OBJ=$(CUSRC:.cu=.o) \
    $(CSRC:.cpp=.o) \
    $(CASMSRC:.casm=.o)
endif

clean:
	rm -f *.o *.cubin *.asm *.cuobj *.cpp1.ii *.cpp4.ii *.cudafe1.* *.fatbin* *.module_id *.ptx
	echo "CUDA_INSTALLED : $(CUDA_INSTALLED)"
	echo "CSRC : $(CSRC)"
	echo "CUSRC : $(CUSRC)"

build : so

cubin : 
	$(NVCC) $(CUBINFLAGS) $(EXTRA_CFLAGS) $(CUSRC) -o $(CUBIN_NAME)
	$(NVDISASM) $(CUBIN_NAME) $(NVDISASM_FLAGS) > $(NVDISASM_FILE)
	$(CUOBJDUMP) $(CUBIN_NAME) $(CUOBJDUMP_FLAGS) > $(CUOBJ_FILE)

##################################################
so: $(OBJ)
ifeq ($(CUDA_INSTALLED),)
	$(CC) $(LINKFLAGS2) $(LIB_PATH)/$(CUSNARKS_LIB) $(OBJ) $(LDFLAGS2)
else
	$(NVCC) $(LINKFLAGS) $(LIB_PATH)/$(CUSNARKS_LIB) $(OBJ) $(LDFLAGS)
endif

##################################################
%.o : %.cu
ifneq ($(CUDA_INSTALLED),)
	$(NVCC) $(CFLAGS) $(EXTRA_CFLAGS) $(CUFLAGS)  $*.cu -o $*.o	
endif

##################################################
log.o : log.cpp
ifneq ($(CUDA_INSTALLED),)
	$(NVCC) $(CFLAGS) $(EXTRA_CFLAGS) -x cu $(CUFLAGS) $*.cpp -o $*.o
else
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS2) $*.cpp -o $*.o	
endif	

%.o : %.cpp
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS2) $*.cpp -o $*.o	

%.o : %.casm
	$(ASM) $(ASMFLAGS) $*.casm -o $*.o	

