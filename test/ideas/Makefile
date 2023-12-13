#
#
# ------------------------------------------------------------------
# Author     : David Ruiz
#
# File name  : Makefile
#
# Date       : 28/01/2020
#
# ------------------------------------------------------------------
#

INCLUDE_PATH = $(INCLUDE_PATH)
LIB_PATH = $(LIB_PATH)
CUSNARKS_LIB=$(CUSNARKS_LIB)

NVCC = nvcc
GENCODE_ARCH_COMPUTE=-gencode arch=compute_61,code=sm_61
CFLAGS=-g -DLINUXINTEL64  -DTEST_MODE -std=c++11 -I $(INCLUDE_PATH) -w
CUFLAGS=-G --compiler-options '-fPIC -Wall -Wno-unused-but-set-variable -Wno-comment' -Xptxas -v -Xptxas -maxrregcount=120 $(GENCODE_ARCH_COMPUTE)

EXTRA_CFLAGS=-I $(AUX_INCLUDES)
LINKFLAGS= -arch=sm_61 -L $(LIB_PATH)
LDFLAGS=-lm -lcudart -lgomp -lcusnarks

CUSRC = test_fft.cu

CUDA_INSTALLED = $(shell which $(NVCC))


NVDISASM_FILE:=$(CUBIN_NAME:.cubin=.asm)
CUOBJ_FILE:=$(CUBIN_NAME:.cubin=.cuobj)


ifeq ($(CUDA_INSTALLED),)
:
else
  EXE= $(CSRC:.cu=.out)
endif

clean:
	rm -f *.o *.out *.cuobj *.cpp1.ii *.cpp4.ii *.cudafe1.* *.fatbin* *.module_id *.ptx

test : $(CUSRC)
	$(NVCC) $(CFLAGS) $(EXTRA_CFLAGS) $(CUFLAGS) $(LINKFLAGS)  test_sc1mul.cu -o test_sc1mul.out $(LDFLAGS)	

