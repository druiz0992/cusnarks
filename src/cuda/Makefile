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
LIB_NAME=$(LIB_NAME)

NVCC = nvcc
GENCODE_ARCH_COMPUTE=-gencode arch=compute_60,code=sm_60
CFLAGS=-c -O2 -DLINUXINTEL64 -g -I $(INCLUDE_PATH) -w
CUFLAGS=--compiler-options '-fPIC -Wall -Wno-unused-but-set-variable -Wno-comment' -Xptxas -dlcm=cg --cudart shared $(GENCODE_ARCH_COMPUTE) 
LIBFLAGS= --shared -o 
LDFLAGS=-lm -lcudart

SRC = bigint.cu \
      bigint_device.cu


OBJ=$(SRC:.cu=.o)

build : so

##################################################
so: $(OBJ)
	$(NVCC) $(LIBFLAGS) $(LIB_PATH)/$(LIB_NAME) $(OBJ) $(LDFLAGS)
	mv $(OBJ) $(OBJECT_PATH)/

##################################################
%.o : %.cu
	$(NVCC) $(CFLAGS) $(CUFLAGS) $*.cu -o $*.o	
