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
//  Implementation of Cusnarks CUDA resources management
//   
// ------------------------------------------------------------------

*/

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <sys.time.h>
#include <cuda_runtime.h>

#include "types.h"
#include "cuda.h"
#include "rng.h"
#include "cusnarks_kernel.h"


using namespace std;

__constant__ mod_info_t mod_info_ct;

/*
    Constructor : Reserves device memory for vector and modulo p. 

    Arguments :
      p : 256 bit number in 8 word uint32 array
      length : Vector length for future arithmetic operations
*/
CUSnarks::CUSnarks (const mod_info_t *mod_info, uint32_t device_vector_len, uint32_t in_size, uint32_t out_size)
{
  CUSnarks(p, device_vector_len, in_size, out_size 0);
}

CUSnarks::CUSnarks (const mod_info_t *mod_info, uint32_t device_vector_len, uint32_t in_size, uint32_t out_size, uint32_t seed) : in_vector_len(device_vector_len)
{
  allocateCudaResources(mod_info, in_size, out_size);
  initRNG(seed);
}

void CUSnarks::allocateCudaResources(const mod_info_t *mod_info, uint32_t in_size, uint32_t out_size)
{
  // Allocate global memory in device for input and output
  CCHECK(cudaMalloc((void**) &this->in_vector_device, in_size));

  CCHECK(cudaMalloc((void**) &this->out_vector_device, out_size);

  // Copy modulo p to device memory
  CCHECK(cudaMemcpytoSymbol(mod_info_ct, mod_info, sizeof(mod_info_t), cudaMemcpyHostToDevice));
}
void CUSnarks::initRNG(uint32_t seed)
{
  if (seed == 0){ rng =  _RNG::get_instance(); }
  else { rng = _RNG::get_instance(seed); }
}

void CUSnarks::rand(uint32_t *samples, uint32_t n_samples, uint32_t size)
{
    rng->randu32(samples, n_samples * size);
}


/*
    Transfer input vector from host to device

    Arguments :
      in_vector_host : Input vector of upto N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      len : number of elements in input vector to be xferred. 
          Cannot be greater than amount reseved during constructor, but not checked
*/
void CUSnarks::copyVectorToDevice(const uint32_t *in_vector_host, uint32_t size)
{
  // Copy input data to device memory
  CCHECK(cudaMemcpy(in_vector_device, in_vector_host, size, cudaMemcpyHostToDevice));
}

/*
    Transfer output vector from device to host

    Arguments :
      out_vector_host : Output vector of upto N/2 256 bit elements Y[0], Y[1], Y[2] ... Y[N/2-1].
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
void CUSnarks::copyVectorFromDevice(uint32_t *out_vector_host, uint32_t out_size)
{
  // copy results from device to host
  CCHECK(cudaMemcpyHostToDeviceMemcpy(out_vector_host, out_vector_device, out_size, cudaMemcpyDeviceToHost));
  CCHECK(cudaGetLastError());
}

CUSnarks::~CUSnarks()
{
  cudaFree(in_vector_device);
  cudaFree(out_vector_device);
}

template<typename kernel_function_t, typename... kernel_parameters_t>
void CUSnarks::kernelLaunch(
		const kernel_function_t& kernel_function,
		uint32_t *out_vector_host,
	       	const uint32_t *in_vector_host,
	        uint32_t in_size,
		uint32_t out_size,
                kernel_config_t *configuration,
		kernel_parameters_t... kernel_extra_params)
{
  if (len > in_vector_len) { return; }

  double start, end_copy_in, end_kernel, end_copy_out;
  int blockD, gridD;

  start = elapsedTime();
  copyVectorToDevice(in_vector_host, in_size);
  end_copy_in = elapsedTime() - start;

  // perform addition operation and leave results in device memory
  start = elapsedTime();
  blockD = configuration->blockD;
  gridD = configuration->gridD;
  kernel_function<<<gridD, blockD>>>(out_vector_device, in_vector_device, len, kernel_params...);
  CCHECK(cudaGetLastError());
  end_kernel = elapsedTime() - start;

  start = elapsedTime();
  CCHECK(cudaDeviceSynchronize());
  copyVectorFromDevice(out_vector_host, out_size);
  end_copy_out = elapsedTime() - start;

  printf("%s <<<%d, %d>>> Time Elapsed Kernel :%f.\n", 
          kernel_function, gridD, blockD, end_kernel);
  printf("Time Elapsed Xfering in %d bytes : %f\n",
          in_size, end_copy_in);
  printf("Time Elapsed Xfering out %d bytes : %f\n",
          out_size, end_copy_out);
}

/*
  Professional CUDA C Programming by John Cheng, Max Grossman, Ty McKercher
*/
double CUSnarks::elapsedTime(void)
{
  struct timeval tp;
  gettimeofday (&tp, NULL);
  return ((double)tp.tv_sec * (double)tp.tv_usec*1e-6);
}


/*
  Professional CUDA C Programming by John Cheng, Max Grossman, Ty McKercher
*/
double CUSnarks::elapsedTime(void)
{
  struct timeval tp;
  gettimeofday (&tp, NULL);
  return ((double)tp.tv_sec * (double)tp.tv_usec*1e-6);
}

/*
  Professional CUDA C Programming by John Cheng, Max Grossman, Ty McKercher
*/
void CUSnarks::getDeviceInfo(void)
{
   int deviceCount = 0;
   CCHECK(cudaGetDeviceCount(&deviceCount));
  
   if (deviceCount == 0) {
      printf("There are no available device(s) that support CUDA\n");
   } else {
      printf("Detected %d CUDA Capable device(s)\n", deviceCount);
   }
   int dev, driverVersion = 0, runtimeVersion = 0;

   dev =0;
   cudaSetDevice(dev);
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, dev);
   printf("Device %d: \"%s\"\n", dev, deviceProp.name);

   cudaDriverGetVersion(&driverVersion);
   cudaRuntimeGetVersion(&runtimeVersion);
   printf(" CUDA Driver Version / Runtime Version                     %d.%d / %d.%d\n",
          driverVersion/1000, (driverVersion%100)/10,
          runtimeVersion/1000, (runtimeVersion%100)/10);
   printf(" CUDA Capability Major/Minor version number:               %d.%d\n",
         deviceProp.major, deviceProp.minor);
   printf(" Total amount of global memory:                            %.2f MBytes (%llu bytes)\n",
         (float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
         (unsigned long long) deviceProp.totalGlobalMem);
   printf(" GPU Clock rate:                                           %.0f MHz (%0.2f GHz)\n",
         deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
   printf(" Memory Clock rate:                                        %.0f Mhz\n",
         deviceProp.memoryClockRate * 1e-3f);
   printf(" Memory Bus Width:                                         %d-bit\n",
         deviceProp.memoryBusWidth);
   if (deviceProp.l2CacheSize) {
         printf(" L2 Cache Size:                                      %d bytes\n",
           deviceProp.l2CacheSize);
   }
   printf(" Max Texture Dimension Size (x,y,z)             "
     "    1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D , deviceProp.maxTexture2D[0],
            deviceProp.maxTexture2D[1], 
            deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
            deviceProp.maxTexture3D[2]);
   printf(" Max Layered Texture Size (dim) x layers
   1D=(%d) x %d, 2D=(%d,%d) x %d\n",
       deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
       deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
       deviceProp.maxTexture2DLayered[2]);

   printf(" Total amount of constant memory:                         %lu bytes\n",
      deviceProp.totalConstMem);
   printf(" Total amount of shared memory per block:                 %lu bytes\n",
      deviceProp.sharedMemPerBlock);
   printf(" Total number of registers available per block:           %d\n",
      deviceProp.regsPerBlock);
   printf(" Warp size:                                               %d\n", deviceProp.warpSize);
   printf(" Maximum number of threads per multiprocessor:            %d\n",
      deviceProp.maxThreadsPerMultiProcessor);
   printf(" Maximum number of threads per block:                     %d\n",
      deviceProp.maxThreadsPerBlock);
   printf(" Maximum sizes of each dimension of a block:              %d x %d x %d\n",
      deviceProp.maxThreadsDim[0],
      deviceProp.maxThreadsDim[1],
      deviceProp.maxThreadsDim[2]);
   printf(" Maximum sizes of each dimension of a grid:               %d x %d x %d\n",
      deviceProp.maxGridSize[0],
      deviceProp.maxGridSize[1],
      deviceProp.maxGridSize[2]);
   printf(" Maximum memory pitch:                                    %lu bytes\n", 
      deviceProp.memPitch);
}
