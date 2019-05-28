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
// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : cusnarks_kernel.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of CUDA resources management. CUSnarks is the base class
// for all CUDA modules. Class provides functionality for GPU memory allocation
//  and deallocation, kernel launching, time measurement, random number generation
//
// ------------------------------------------------------------------
*/

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "types.h"
#include "constants.h"
#include "cuda.h"
#include "log.h"
//#include "rng.h"
#include "utils_host.h"
#include "cusnarks_kernel.h"


using namespace std;


// Prime information for finitie fields. Includes 3 numbers : p. p_ and r_ that 
// follow p x p_ - r * r_ = 1 whrere r is 1^256. This is used for Montgomery reduction
//
// There are two different set of primes (MOD_N)
__constant__ mod_info_t mod_info_ct[MOD_N];

// EC BN128 curve and params definition
// Y^2 = X^3 + b
// b = 3
// Generator point G =(Gx, Gy) = (1,2)
// Generator point G2 ([Gx1, Gx2], [Gy1, Gy2])
//       'G1x' : 10857046999023057135944570762232829481370756359578518086990519993285655852781L,
//       'G2x' : 8495653923123431417604973247489272438418190587263600148770280649306958101930L,
//       'G1y' : 11559732032986387107991004021392285783925812861821192530917403151452391805634L,
//       'G2y' : 4082367875863433681332203403145435568316851327593401208105741076214120093531L
//
// Assumption is that we are woking in Mongtgomery domain, so I need to transforms all these parameters
// Also, these parameters will vary depending on prime number used. 

// There are two different set of primes (MOD_N)
__constant__ ecbn128_t ecbn128_params_ct[MOD_N];

// Additional constants
__constant__ misc_const_t misc_const_ct[MOD_N];

// 32 roots of unitity of field prime (only first 16)

__constant__ uint32_t W32_ct[NWORDS_256BIT * 16];

// 32 inverse roots of unitity of field prime (only first 16)

__constant__ uint32_t IW32_ct[NWORDS_256BIT * 16];

// During IFFT, I need to scale by inv(32). Below is the representation of 32 in Mongtgomery

__constant__ uint32_t IW32_nroots_ct[NWORDS_256BIT * (FFT_SIZE_N - 1)];

/*
    Constructor : Reserves global (vector) and constant (prime info) memory 
    Arguments :
      in_length : Maximum number of elements in Kernel input data
      in_size   : Maximum size of Kernel input data (in Bytes)
      out_length: Maximum number of elements in Kernel output  data
      out_size  : Mamimum size of Kernel output data (in Bytes)
      kcb       : Pointer to kernel functions (indexed by XXX_callback_t enum)
      
*/
CUSnarks::CUSnarks (uint32_t in_len, uint32_t in_size, 
                    uint32_t out_len, uint32_t out_size, kernel_cb *kcb) : 
                          kernel_callbacks(kcb)
{
  CUSnarks(in_len, in_size, out_len, out_size, 0);
}

CUSnarks::CUSnarks (uint32_t in_len, uint32_t in_size,
                    uint32_t out_len, uint32_t out_size,
                    kernel_cb *kcb,uint32_t seed) : 
                       kernel_callbacks(kcb)
{
  in_vector_device.data = NULL;
  in_vector_device.length = in_len;
  in_vector_device.size = in_size;
  out_vector_device.data = NULL;
  out_vector_device.length = out_len;
  out_vector_device.size = out_size;

  allocateCudaResources(in_size, out_size);
  //initRNG(seed);
}

/*
   Reserve GPU memory for input and output vectors and input kernel params (global memory),
   as well as some constant info (constant memory)
 */
void CUSnarks::allocateCudaResources(uint32_t in_size, uint32_t out_size)
{
  // Allocate kernel input and putput data vectors in global memory 
  CCHECK(cudaMalloc((void**) &this->in_vector_device.data, in_size));
  CCHECK(cudaMalloc((void**) &this->out_vector_device.data, out_size));

  // Allocate kernel params in global memory 
  CCHECK(cudaMalloc((void**) &this->params_device, sizeof(kernel_params_t)));

  // Copy modulo info to device constant
  CCHECK(cudaMemcpyToSymbol(mod_info_ct,       CusnarksModInfoGet(),     MOD_N * sizeof(mod_info_t)));  // prime info
  CCHECK(cudaMemcpyToSymbol(ecbn128_params_ct, CusnarksEcbn128ParamsGet(), MOD_N * sizeof(ecbn128_t)));   // ecbn128
  CCHECK(cudaMemcpyToSymbol(misc_const_ct,    CusnarksMiscKGet(),    MOD_N * sizeof(misc_const_t)));// misc
  CCHECK(cudaMemcpyToSymbol(W32_ct,           CusnarksW32RootsGet(), sizeof(uint32_t) * NWORDS_256BIT * 16));// W32roots
  CCHECK(cudaMemcpyToSymbol(IW32_ct,          CusnarksIW32RootsGet(), sizeof(uint32_t) * NWORDS_256BIT * 16));// IW32roots
  CCHECK(cudaMemcpyToSymbol(IW32_nroots_ct,   CusnarksIW32NRootsGet(), sizeof(uint32_t) * NWORDS_256BIT * (FFT_SIZE_N -1) ));// inverse 2,4,8,16,32
}

/*
   Initialize PCG random number generator  
   http://www.pcg-random.org/
   IF seed is 0, random seed is taken from urand.
   NOTE : when seed is 0, generator breaks
   TODO :  Fix seed = 0
*/
#if 0
void CUSnarks::initRNG(uint32_t seed)
{
  if (seed == 0){ rng =  _RNG::get_instance(); }
  else { rng = _RNG::get_instance(seed); }
}
#endif
/*
   Generate N 32 bit random samples
*/
void CUSnarks::rand(uint32_t *samples, uint32_t n_samples)
{
    uint32_t size_sample = in_vector_device.size / (in_vector_device.length * sizeof(uint32_t));
    //rng->randu32(samples, n_samples * size_sample);
    setRandom(samples, n_samples * size_sample);
}

void CUSnarks::randu256(uint32_t *samples, uint32_t n_samples, uint32_t *mod=NULL)
{
    //uint32_t size_sample = in_vector_device.size / (in_vector_device.length * sizeof(uint32_t));
    //rng->randu256(samples, n_samples * size_sample, 1);
    //rng->randu256(samples, n_samples, mod);
    setRandom256(samples, n_samples, mod);
}

void CUSnarks::saveFile(uint32_t *samples, uint32_t n_samples, char *fname)
{
  uint32_t i;
  FILE *pFile;

  pFile = fopen(fname,"wb");

  for (i=0; i< n_samples * 8; i++){
    fwrite(&samples[i],sizeof(uint32_t), 1, pFile);
  }
  fclose(pFile);
}

/*
   Free memory allocated in GPU:
*/
CUSnarks::~CUSnarks()
{
  logInfo("Release resources\n");
  cudaFree(in_vector_device.data);
  cudaFree(out_vector_device.data);
  cudaFree(params_device);
}

/*
   Kernel launcher. This function is an attempt to hide the complexity of launching a kernel. When called,
   the input vector is copied to global GPU memory, the kernel is launched, and when finished, kernel
   output vector data is copied back from GPU to host.
   Arguments:
    kernel_idx : kernel number to be launched. Defined by XXX_callback_t enum types
    out_vector_host : kernel ouput data vector (Host size)
    in_vector_host  : kernel input data vector (host size)
    config          : kernel configuration info (grid, block, smem,...)
    params          : Kernel input parameters
*/
double CUSnarks::kernelLaunch(
		vector_t *out_vector_host,
	       	vector_t *in_vector_host,
                kernel_config_t *config,
                kernel_params_t *params,
                uint32_t n_kernel=1)
{
  uint32_t i;
  // check input lengths do not exceed reserved amount
  if (in_vector_host->length > in_vector_device.length) { 
    logInfo("Error IVHL : %d >  IVDL : %d\n",in_vector_host->length, in_vector_device.length);
    return 0.0;
  }
  if (out_vector_host->length > out_vector_device.length) {
    logInfo("Error OVHL : %d > OVDL : %d\n",out_vector_host->length, out_vector_device.length);
    return 0.0;
  }

  in_vector_host->size = in_vector_host->length * (in_vector_device.size / in_vector_device.length  );
  out_vector_host->size = out_vector_host->length * (out_vector_device.size / out_vector_device.length );

  double start, end_copy_in, end_kernel, end_copy_out, total_kernel;
  int blockD, gridD, smemS, kernel_idx;

  // measure data xfer time Host -> Device
  start = elapsedTime();
  //printf("%d. %d, %d\n", config[0].in_offset, in_vector_host->data, in_vector_host->size);
  CCHECK(cudaMemcpy(&in_vector_device.data[config[0].in_offset], in_vector_host->data, in_vector_host->size, cudaMemcpyHostToDevice));
  end_copy_in = elapsedTime() - start;

  total_kernel = 0.0;
 
  // configure kernel. Input parameter invludes block size. Grid is calculated 
  // depending on input data length and stride (how many samples of input data are 
  // used per thread
  for (i=0; i < n_kernel; i++){
    start = elapsedTime();
    CCHECK(cudaMemcpy(params_device, &params[i], sizeof(kernel_params_t), cudaMemcpyHostToDevice));
    end_copy_in = +(elapsedTime() - start);
    blockD = config[i].blockD;
    if (config[i].gridD == 0){
       config[i].gridD = (blockD + in_vector_host->length/params[i].stride - 1) / blockD;
    }
    gridD = config[i].gridD;
    smemS = config[i].smemS;
    kernel_idx = config[i].kernel_idx;


    // launch kernel
    start = elapsedTime();
    kernel_callbacks[kernel_idx]<<<gridD, blockD, smemS>>>(out_vector_device.data, in_vector_device.data, params_device);
    CCHECK(cudaGetLastError());
    CCHECK(cudaDeviceSynchronize());
    end_kernel = elapsedTime() - start;
    total_kernel +=end_kernel;

    logInfo("Params : premod : %d, midx : %d, In Length : %d, Out Length : %d, Stride : %d, Padding Idx : %d\n",
        params[i].premod, params[i].midx, params[i].in_length, params[i].out_length, params[i].stride, params[i].padding_idx);
    logInfo("Kernel IDX :%d <<<%d, %d, %d>>> Time Elapsed Kernel : %f.sec\n", 
          kernel_idx, gridD, blockD, smemS,end_kernel);
  }
  
    // retrieve kernel output data from GPU to host
  start = elapsedTime();
  if (config[0].return_val){
     CCHECK(cudaMemcpy(out_vector_host->data, out_vector_device.data, out_vector_host->size, cudaMemcpyDeviceToHost));
  }
 
  end_copy_out = elapsedTime() - start;

  logInfo("----- Info -------\n");
  logInfo("IVHS : %d, IVHL : %d, IVDS : %d, IVDL : %d\n",in_vector_host->size, 
		                                        in_vector_host->length,
						       	in_vector_device.size,
						       	in_vector_device.length);

  logInfo("OVHS : %d, OVHL : %d, OVDS : %d, OVDL : %d\n",out_vector_host->size,
		                                        out_vector_host->length, 
							out_vector_device.size,
						       	out_vector_device.length);

  logInfo("Time Elapsed Xfering in %d bytes : %f sec\n",
          in_vector_host->size, end_copy_in);
  logInfo("Time Elapsed Xfering out %d bytes : %f sec\n",
          out_vector_host->size, end_copy_out);

  return total_kernel;
}


/*
  Professional CUDA C Programming by John Cheng, Max Grossman, Ty McKercher
*/
double CUSnarks::elapsedTime(void)
{
  struct timeval tp;
  gettimeofday (&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

/*
  Professional CUDA C Programming by John Cheng, Max Grossman, Ty McKercher
*/
void CUSnarks::getDeviceInfo(void)
{
   int deviceCount = 0;
   CCHECK(cudaGetDeviceCount(&deviceCount));
  
   if (deviceCount == 0) {
      logInfo("There are no available device(s) that support CUDA\n");
   } else {
      logInfo("Detected %d CUDA Capable device(s)\n", deviceCount);
   }
   int dev, driverVersion = 0, runtimeVersion = 0;

   dev =0;
   cudaSetDevice(dev);
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, dev);
   logInfo("Device %d: \"%s\"\n", dev, deviceProp.name);

   cudaDriverGetVersion(&driverVersion);
   cudaRuntimeGetVersion(&runtimeVersion);
   logInfo(" CUDA Driver Version / Runtime Version                     %d.%d / %d.%d\n",
          driverVersion/1000, (driverVersion%100)/10,
          runtimeVersion/1000, (runtimeVersion%100)/10);
   logInfo(" CUDA Capability Major/Minor version number:               %d.%d\n",
         deviceProp.major, deviceProp.minor);
   logInfo(" Total amount of global memory:                            %.2f MBytes (%llu bytes)\n",
         (float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
         (unsigned long long) deviceProp.totalGlobalMem);
   logInfo(" GPU Clock rate:                                           %.0f MHz (%0.2f GHz)\n",
         deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
   logInfo(" Memory Clock rate:                                        %.0f Mhz\n",
         deviceProp.memoryClockRate * 1e-3f);
   logInfo(" Memory Bus Width:                                         %d-bit\n",
         deviceProp.memoryBusWidth);
   if (deviceProp.l2CacheSize) {
         logInfo(" L2 Cache Size:                                      %d bytes\n",
           deviceProp.l2CacheSize);
   }
   logInfo(" Max Texture Dimension Size (x,y,z)             "
     "    1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D , deviceProp.maxTexture2D[0],
            deviceProp.maxTexture2D[1], 
            deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
            deviceProp.maxTexture3D[2]);
   logInfo(" Max Layered Texture Size (dim) x layers         "
     "   1D=(%d) x %d, 2D=(%d,%d) x %d\n",
       deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
       deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
       deviceProp.maxTexture2DLayered[2]);

   logInfo(" Total amount of constant memory:                         %lu bytes\n",
      deviceProp.totalConstMem);
   logInfo(" Total amount of shared memory per block:                 %lu bytes\n",
      deviceProp.sharedMemPerBlock);
   logInfo(" Total number of registers available per block:           %d\n",
      deviceProp.regsPerBlock);
   logInfo(" Warp size:                                               %d\n", deviceProp.warpSize);
   logInfo(" Maximum number of threads per multiprocessor:            %d\n",
      deviceProp.maxThreadsPerMultiProcessor);
   logInfo(" Maximum number of threads per block:                     %d\n",
      deviceProp.maxThreadsPerBlock);
   logInfo(" Maximum sizes of each dimension of a block:              %d x %d x %d\n",
      deviceProp.maxThreadsDim[0],
      deviceProp.maxThreadsDim[1],
      deviceProp.maxThreadsDim[2]);
   logInfo(" Maximum sizes of each dimension of a grid:               %d x %d x %d\n",
      deviceProp.maxGridSize[0],
      deviceProp.maxGridSize[1],
      deviceProp.maxGridSize[2]);
   logInfo(" Maximum memory pitch:                                    %lu bytes\n", 
      deviceProp.memPitch);
}
