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
#include <sys/time.h>
#include <cuda_runtime.h>

#include "types.h"
#include "cuda.h"
#include "rng.h"
#include "cusnarks_kernel.h"


using namespace std;

__constant__ mod_info_t mod_info_ct[MOD_N];

//  p1 = 21888242871839275222246405745257275088696311157297823662689037894645226208583L
// 'Pp': 111032442853175714102588374283752698368366046808579839647964533820976443843465L,
// 'R': 115792089237316195423570985008687907853269984665640564039457584007913129639936L,
// 'R3modP': 14921786541159648185948152738563080959093619838510245177710943249661917737183L,
// 'Rbitlen': 256,
// 'Rmask': 115792089237316195423570985008687907853269984665640564039457584007913129639935L,
// 'RmodP': 6350874878119819312338956282401532409788428879151445726012394534686998597021L,
// 'Rp': 20988524275117001072002809824448087578619730785600314334253784976379291040311

static uint32_t p1_u256[]  = {3632069959, 1008765974, 1752287885, 2541841041, 
                              2172737629, 3092268470, 3778125865,  811880050};
static uint32_t p1n_u256[] = {3834012553, 2278688642,  516582089, 2665381221, 
                               406051456, 3635399632, 2441645163, 4118422199};
static uint32_t r1n_u256[] = {21690935,   3984885834,   41479672, 3944751749,
                              3074724569, 3479431631, 1508230713,  778507633};

//p2 = 21888242871839275222246405745257275088548364400416034343698204186575808495617L
// 'Pp': 52454480824480482120356829342366457550537710351690908576382634413609933864959L,
// 'R': 115792089237316195423570985008687907853269984665640564039457584007913129639936L,
// 'R3modP': 5866548545943845227489894872040244720403868105578784105281690076696998248512L,
// 'Rbitlen': 256,
// 'Rmask': 115792089237316195423570985008687907853269984665640564039457584007913129639935L,
// 'RmodP': 6350874878119819312338956282401532410528162663560392320966563075034087161851L,
// 'Rp': 9915499612839321149637521777990102151350674507940716049588462388200839649614L}


static uint32_t p2_u256[]  = {4026531841, 1138881939, 2042196113,  674490440,
                              2172737629, 3092268470, 3778125865,  811880050};
static uint32_t p2n_u256[] = {4026531839, 3269588371, 1281954227,  1703315019,
                              2567316369, 3818559528,  226705842,  1945644829};
static uint32_t r2n_u256[] = {1840322894, 3696992261, 3776048263,   151975337,
                              2931318109, 3357937124, 2193970460,   367786321};

/*
    Constructor : Reserves device memory for vector and modulo p. 

    Arguments :
      p : 256 bit number in 8 word uint32 array
      length : Vector length for future arithmetic operations
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
  printf("C CUSnarks init :  in_size %d, in_len %d\n", in_vector_device.size, in_vector_device.length);
  printf("C CUSnarks init :  out_size %d, out_len %d\n", out_vector_device.size, out_vector_device.length);

  allocateCudaResources(in_size, out_size);
  initRNG(seed);
}

void CUSnarks::printBigNumber(uint32_t *x)
{
  for (int i=0; i< NWORDS_256BIT; i++){
    printf("%u ",x[i]);
  }
  printf("\n");
}

void CUSnarks::allocateCudaResources(uint32_t in_size, uint32_t out_size)
{
  mod_info_t mod_h[MOD_N];

  // Allocate global memory in device for input and output
  CCHECK(cudaMalloc((void**) &this->in_vector_device.data, in_size));

  CCHECK(cudaMalloc((void**) &this->out_vector_device.data, out_size));

  CCHECK(cudaMalloc((void**) &this->params_device, sizeof(kernel_params_t)));

  memcpy(&mod_h[MOD_GROUP].p,  p1_u256,  sizeof(uint32_t) * NWORDS_256BIT);
  memcpy(&mod_h[MOD_GROUP].p_, p1n_u256, sizeof(uint32_t) * NWORDS_256BIT);
  memcpy(&mod_h[MOD_GROUP].r_, r1n_u256, sizeof(uint32_t) * NWORDS_256BIT);
  memcpy(&mod_h[MOD_FIELD].p,  p2_u256,  sizeof(uint32_t) * NWORDS_256BIT);
  memcpy(&mod_h[MOD_FIELD].p_, p2n_u256, sizeof(uint32_t) * NWORDS_256BIT);
  memcpy(&mod_h[MOD_FIELD].r_, r2n_u256, sizeof(uint32_t) * NWORDS_256BIT);

  printBigNumber(mod_h[MOD_GROUP].p);
  printBigNumber(mod_h[MOD_GROUP].p_);
  printBigNumber(mod_h[MOD_GROUP].r_);
  printBigNumber(mod_h[MOD_FIELD].p);
  printBigNumber(mod_h[MOD_FIELD].p_);
  printBigNumber(mod_h[MOD_FIELD].r_);

  // Copy modulo info to device constant
  CCHECK(cudaMemcpyToSymbol(mod_info_ct, mod_h, MOD_N * sizeof(mod_info_t)));
}
void CUSnarks::initRNG(uint32_t seed)
{
  if (seed == 0){ rng =  _RNG::get_instance(); }
  else { rng = _RNG::get_instance(seed); }
}

void CUSnarks::rand(uint32_t *samples, uint32_t n_samples)
{
    uint32_t size_sample = in_vector_device.size / (in_vector_device.length * sizeof(uint32_t));
    printf("C rand: size_sampled %d, in_size %d, in_len %d\n",size_sample, in_vector_device.size, in_vector_device.length);
    rng->randu32(samples, n_samples * size_sample);
}


CUSnarks::~CUSnarks()
{
  cudaFree(in_vector_device.data);
  cudaFree(out_vector_device.data);
  cudaFree(params_device);
}

#if 0
template<typename kernel_function_t, typename... kernel_parameters_t>
void CUSnarks::kernelLaunch(
		const kernel_function_t& kernel_function,
		uint32_t *out_vector_host,
	       	const uint32_t *in_vector_host,
                uint32_t len,
	        uint32_t in_size,
		uint32_t out_size,
                kernel_config_t *configuration,
		kernel_parameters_t... kernel_extra_params)
#endif
void CUSnarks::kernelLaunch(
                uint32_t kernel_idx,
		//kernel_cb launcher,
		//void *launcher,
		vector_t *out_vector_host,
	       	vector_t *in_vector_host,
                kernel_config_t *config,
                kernel_params_t *params)
{
  if (in_vector_host->length > in_vector_device.length) { return; }
  //if (in_vector_host->size > in_vector_device.size) { return; }
  if (out_vector_host->length > out_vector_device.length) { return; }
  //if (out_vector_host->size > out_vector_device.size) { return; }

  in_vector_host->size = in_vector_host->length * (in_vector_device.size / in_vector_device.length  );
  out_vector_host->size = out_vector_host->length * (out_vector_device.size / out_vector_device.length );

  printf("IVHS : %d, IVHL : %d, IVDS : %d, IDDL : %d\n",in_vector_host->size, in_vector_host->length, in_vector_device.size, in_vector_device.length);
  printf("OVHS : %d, OVHL : %d, OVDS : %d, ODDL : %d\n",out_vector_host->size, out_vector_host->length, out_vector_device.size, out_vector_device.length);

  double start, end_copy_in, end_kernel, end_copy_out;
  int blockD, gridD;

  // measure xfer time Host -> Device
  start = elapsedTime();
  CCHECK(cudaMemcpy(in_vector_device.data, in_vector_host->data, in_vector_host->size, cudaMemcpyHostToDevice));
  CCHECK(cudaMemcpy(params_device, params, sizeof(kernel_params_t), cudaMemcpyHostToDevice));
  end_copy_in = elapsedTime() - start;

  for (int i=0; i< 10; i++){
    printf("%d-%d\n",in_vector_host->data[i], out_vector_host->data[i]);
  }

  blockD = config->blockD;
  config->gridD = (blockD + in_vector_host->length/params->stride - 1) / blockD;
  gridD = config->gridD;

  printf("Config : blockD: %d, gridD: %d\n",config->blockD, config->gridD);
  printf("Params : premod : %d, midx : %d, Length : %d\n",params->premod, params->midx, params->length);

  // perform addition operation and leave results in device memory
  start = elapsedTime();
  printf("CB[%d] : %pF at address %p\n",kernel_idx,kernel_callbacks[kernel_idx], kernel_callbacks[kernel_idx]);
  kernel_callbacks[kernel_idx]<<<gridD, blockD>>>(out_vector_device.data, in_vector_device.data, params_device);
  //kernel_callbacks[kernel_idx](&out_vector_device, &in_vector_device, params);
  CCHECK(cudaGetLastError());
  CCHECK(cudaDeviceSynchronize());
  end_kernel = elapsedTime() - start;

  start = elapsedTime();
  CCHECK(cudaMemcpy(out_vector_host->data, out_vector_device.data, out_vector_host->size, cudaMemcpyDeviceToHost));
  end_copy_out = elapsedTime() - start;

  printf("%pF <<<%d, %d>>> Time Elapsed Kernel : %f.sec\n", 
          kernel_callbacks[kernel_idx], gridD, blockD, end_kernel);
  printf("Time Elapsed Xfering in %d bytes : %f sec\n",
          in_vector_host->size, end_copy_in);
  printf("Time Elapsed Xfering out %d bytes : %f sec\n",
          out_vector_host->size, end_copy_out);
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
   printf(" Max Layered Texture Size (dim) x layers         "
     "   1D=(%d) x %d, 2D=(%d,%d) x %d\n",
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
