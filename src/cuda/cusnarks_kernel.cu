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


static  int deviceCount = 0;

uint32_t CUSnarks::init_resources=0;
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

  if (CUSnarks::init_resources==0){
    CUSnarks::init_resources = 1;

    // initialize device count
    CCHECK(cudaGetDeviceCount(&deviceCount));
    // reset devices
    resetDevices();
    // alocate constant mem objects
    allocateCudaCteResources();
  }
  //allocate streams and events
  allocateCudaStreamResources();
  // alocate global mem objects
  allocateCudaResources(in_size, out_size, in_len, out_len);
}

void CUSnarks::resetDevices(void)
{
   uint32_t i;
   for(i=0; i< deviceCount; i++){
    CCHECK(cudaSetDevice(i));
    CCHECK(cudaDeviceReset());
   }
}

/*
   Reserve streams, events,...
 */
void CUSnarks::allocateCudaStreamResources(void)
{
  uint32_t i,j;

  // create vectors for all devices
  stream = (cudaStream_t **)malloc(deviceCount * sizeof(cudaStream_t *));
  start_event = (cudaEvent_t **)malloc(deviceCount * sizeof(cudaEvent_t *));
  end_event = (cudaEvent_t **)malloc(deviceCount * sizeof(cudaEvent_t *));

  if ((stream== NULL) || 
       (start_event == NULL) || (end_event == NULL) ) { 
    logInfo("Cannot allocate memory. Exiting program...\n");
    exit(1);
  }
  
  for (i=0; i < deviceCount; i++){
    CCHECK(cudaSetDevice(i));

    stream[i] = (cudaStream_t *)malloc(N_STREAMS_PER_GPU * sizeof(cudaStream_t ));
    start_event[i] = (cudaEvent_t *)malloc(N_STREAMS_PER_GPU * sizeof(cudaEvent_t ));
    end_event[i] = (cudaEvent_t *)malloc(N_STREAMS_PER_GPU * sizeof(cudaEvent_t ));

    if ((stream[i] == NULL) || (start_event[i]==NULL) || (end_event[i]==NULL)){
      logInfo("Cannot allocate memory. Exiting program...\n");
      exit(1);
    }
     
    for (j=0; j < N_STREAMS_PER_GPU; j++){
       CCHECK(cudaStreamCreate(&stream[i][j]));
       CCHECK(cudaEventCreate(&start_event[i][j]));
       CCHECK(cudaEventCreate(&end_event[i][j]));
    }
  }
}



/*
   Reserve GPU memory for input and output vectors and input kernel params (global memory),
 */
void CUSnarks::allocateCudaResources(uint32_t in_size, uint32_t out_size, uint32_t in_len, uint32_t out_len)
{
  uint32_t i,j;

  // Init resources in device. Do it for all GPUs:
  //  - We will use all GPUs for every stage, so it is not a waste
  //  - I have to change a lot of code to allow to specify destinaton GPU
  // Allocate kernel input and putput data vectors in global memory 

  // create vectors for all devices
  in_vector_device = (vector_t **)malloc(deviceCount * sizeof(vector_t *));
  out_vector_device = (vector_t **)malloc(deviceCount * sizeof(vector_t *));
  params_device = (kernel_params_t ***)malloc(deviceCount * sizeof(kernel_params_t **));


  in_data_host = (uint32_t ***)malloc(deviceCount * sizeof(uint32_t **));
  out_data_host = (uint32_t ***)malloc(deviceCount * sizeof(uint32_t **));
  params_host = (kernel_params_t ***)malloc(deviceCount * sizeof(kernel_params_t **));


  if ((in_vector_device == NULL) || (in_data_host == NULL) ||
       (out_vector_device == NULL) || (out_data_host == NULL) ||
       (params_device == NULL) || (params_host == NULL)){
    logInfo("Cannot allocate memory. Exiting program...\n");
    exit(1);
  }
  
  for (i=0; i < deviceCount; i++){

    CCHECK(cudaSetDevice(i));

    in_vector_device[i] = (vector_t *)malloc(N_STREAMS_PER_GPU * sizeof(vector_t));
    out_vector_device[i] = (vector_t *)malloc(N_STREAMS_PER_GPU * sizeof(vector_t));
    params_device[i] = (kernel_params_t **)malloc(N_STREAMS_PER_GPU * sizeof(kernel_params_t *));
  

    out_data_host[i] = (uint32_t **)malloc(N_STREAMS_PER_GPU * sizeof(uint32_t *));
    in_data_host[i] = (uint32_t **)malloc(N_STREAMS_PER_GPU * sizeof(uint32_t *));
    params_host[i] = (kernel_params_t **)malloc(N_STREAMS_PER_GPU * sizeof(kernel_params_t *));

    if ((in_vector_device[i] == NULL) || (in_data_host[i] == NULL) ||
       (out_vector_device[i] == NULL) || (out_data_host[i] == NULL) ||
       (params_device[i] == NULL) || (params_host[i] == NULL)){
        logInfo("Cannot allocate memory. Exiting program...\n");
         exit(1);
    }

    for (j=0; j < N_STREAMS_PER_GPU; j++){
      // initialize vectors in host mem
      in_vector_device[i][j].data = NULL;
      in_vector_device[i][j].length = in_len;
      in_vector_device[i][j].size = in_size;


      out_vector_device[i][j].data = NULL;
      out_vector_device[i][j].length = out_len;
      out_vector_device[i][j].size = out_size;


      // create buffer in device
      CCHECK(cudaMalloc((void**) &this->in_vector_device[i][j].data, in_size));
      CCHECK(cudaMalloc((void**) &this->out_vector_device[i][j].data, out_size));

      // Allocate kernel params in global memory 
      CCHECK(cudaMalloc((void**) &this->params_device[i][j], sizeof(kernel_params_t)));
      logInfo("in size (%d-%d) : %d, data  in: %x, data out : %x, params : %x \n",
          i,j,in_size,
          this->in_vector_device[i][j].data,
          this->out_vector_device[i][j].data,
          this->params_device[i][j]);
    }
  }
}

void CUSnarks::allocateCudaCteResources()
{
  // init constant memory in all GPUS. Do only once, when 
  //  CUsnarks oject is created
  // Copy modulo info to device constant
  uint32_t i;

  for (i=0; i < deviceCount; i++){
    CCHECK(cudaSetDevice(i));


    CCHECK(cudaMemcpyToSymbol(mod_info_ct,       CusnarksModInfoGet(),     MOD_N * sizeof(mod_info_t)));  // prime info
    CCHECK(cudaMemcpyToSymbol(ecbn128_params_ct, CusnarksEcbn128ParamsGet(), MOD_N * sizeof(ecbn128_t)));   // ecbn128
    CCHECK(cudaMemcpyToSymbol(misc_const_ct,    CusnarksMiscKGet(),    MOD_N * sizeof(misc_const_t)));// misc
    CCHECK(cudaMemcpyToSymbol(W32_ct,           CusnarksW32RootsGet(), sizeof(uint32_t) * NWORDS_256BIT * 16));// W32roots
    CCHECK(cudaMemcpyToSymbol(IW32_ct,          CusnarksIW32RootsGet(), sizeof(uint32_t) * NWORDS_256BIT * 16));// IW32roots
    CCHECK(cudaMemcpyToSymbol(IW32_nroots_ct,   CusnarksIW32NRootsGet(), sizeof(uint32_t) * NWORDS_256BIT * (FFT_SIZE_N -1) ));// inverse 2,4,8,16,32
  }
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
     uint32_t size_sample;
    
     size_sample = in_vector_device[0][0].size / (in_vector_device[0][0].length * sizeof(uint32_t));
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
  if (CUSnarks::init_resources){

     CUSnarks::init_resources = 0;
  }
 
  releaseCudaStreamResources();
  releaseCudaResources();
}

void CUSnarks::releaseCudaStreamResources(void)
{
  int i,j;
  /*
    logInfo("Release common resources\n");
  */ 
  for (i=0; i < deviceCount; i++){
      CCHECK(cudaSetDevice(i));
    
      for (j=0; j < N_STREAMS_PER_GPU; j++){
        CCHECK(cudaStreamDestroy(stream[i][j]));
        CCHECK(cudaEventDestroy(start_event[i][j]));
        CCHECK(cudaEventDestroy(end_event[i][j]));
      }

      free(stream[i]);
      free(start_event[i]);
      free(end_event[i]);
  }

  free(stream);
  free(start_event);
  free(end_event);
}

void CUSnarks::releaseCudaResources(void)
{
  int i,j;
  /*
  logInfo("Release Var resources\n");
  */
  
  for (i==0; i < deviceCount; i++){
    
      free(in_vector_device[i]);
      free(out_vector_device[i]);
      free(params_device[i]);

      free(out_data_host[i]);
      free(in_data_host[i]);
      free(params_host[i]);
  }

  free(in_vector_device);
  free(out_vector_device);
  free(params_device);

  free(out_data_host);
  free(in_data_host);
  free(params_host);
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
    gpu_id          : GPU ID
    stream_id	    : stream ID
    n_kernel        : Number of kernels
*/
double CUSnarks::kernelLaunch(
		vector_t *out_vector_host,
	       	vector_t *in_vector_host,
                kernel_config_t *config,
                kernel_params_t *params,
                uint32_t gpu_id=0,
                uint32_t stream_id=0,
                uint32_t n_kernel=1)
{
  uint32_t i;
  
  // check input lengths do not exceed reserved amount
  if (in_vector_host->length > in_vector_device[gpu_id][stream_id].length) { 
    logInfo("Error IVHL : %d >  IVDL : %d\n",in_vector_host->length, in_vector_device[gpu_id][stream_id].length);
    assert(0);
    return 0.0;
  }
  if (out_vector_host->length > out_vector_device[gpu_id][stream_id].length) {
    logInfo("Error OVHL : %d > OVDL : %d\n",out_vector_host->length, out_vector_device[gpu_id][stream_id].length);
    assert(0);
    return 0.0;
  }

  in_vector_host->size = 
        in_vector_host->length *
                 (in_vector_device[gpu_id][stream_id].size / in_vector_device[gpu_id][stream_id].length  );
  out_vector_host->size =
        out_vector_host->length *
               (out_vector_device[gpu_id][stream_id].size / out_vector_device[gpu_id][stream_id].length );

  double end_copy_in=0.0, end_copy_out=0.0, total_kernel=0.0;
  double start_copy_in, start_kernel, end_kernel,start_copy_out;

  int blockD, gridD, smemS, kernel_idx;

  // measure data xfer time Host -> Device
  cudaSetDevice(gpu_id);
  double _start, _end, _total_start, _total_end;
  _total_start = elapsedTime();
  _start = elapsedTime();
  if (stream_id == 0){
     start_copy_in = elapsedTime();

     CCHECK(cudaMemcpy(
              &in_vector_device[gpu_id][stream_id].data[config[0].in_offset],
              in_vector_host->data,
              in_vector_host->size,
              cudaMemcpyHostToDevice));
     end_copy_in = elapsedTime() - start_copy_in;
  } else {
    in_data_host[gpu_id][stream_id] = in_vector_host->data;
    CCHECK(cudaMemcpyAsync(
             &in_vector_device[gpu_id][stream_id].data[config[0].in_offset],
             in_vector_host->data,
             in_vector_host->size,
             cudaMemcpyHostToDevice,
             this->stream[gpu_id][stream_id]));
    _end = elapsedTime() - _start;
    logInfo("In Data : Ptr : %x, gpu_id : %u, stream_id : %u, Time : %f\n",
           in_data_host[gpu_id][stream_id], gpu_id, stream_id, _end);
  }
 
  // configure kernel. Input parameter invludes block size. Grid is calculated 
  // depending on input data length and stride (how many samples of input data are 
  // used per thread
  _start = elapsedTime();
  params_host[gpu_id][stream_id]= (kernel_params_t *) params;
  for (i=0; i < n_kernel; i++){
    if (stream_id == 0){
      start_copy_in = elapsedTime();
      CCHECK(cudaMemcpy(params_device[gpu_id][stream_id],
                        &params[i],
                        sizeof(kernel_params_t),
                        cudaMemcpyHostToDevice));
      end_copy_in = +(elapsedTime() - start_copy_in);
    } else {
      CCHECK(cudaMemcpyAsync(params_device[gpu_id][stream_id],
                        &params_host[gpu_id][stream_id][i],
                        sizeof(kernel_params_t),
                        cudaMemcpyHostToDevice,
                        this->stream[gpu_id][stream_id]));
      _end = elapsedTime() - _start;
       logInfo("In Params : Ptr : %x, gpu_id : %u, stream_id : %u, time : %f\n",
           params_host[gpu_id][stream_id], gpu_id, stream_id, _end);
    }
    blockD = config[i].blockD;
    if (config[i].gridD == 0){
       config[i].gridD = (blockD + in_vector_host->length/params[i].stride - 1) / blockD;
    }
    gridD = config[i].gridD;
    smemS = config[i].smemS;
    kernel_idx = config[i].kernel_idx;

    _start = elapsedTime();
    // launch kernel
    if (stream_id == 0){
       start_kernel = elapsedTime();
       kernel_callbacks[kernel_idx]<<<gridD, blockD, smemS>>>(
                                                  out_vector_device[gpu_id][stream_id].data,
                                                  in_vector_device[gpu_id][stream_id].data,
                                                   params_device[gpu_id][stream_id]
                                                             );
       end_kernel = elapsedTime() - start_kernel;
       total_kernel +=end_kernel;
    } else {
       CCHECK(cudaEventRecord(start_event[gpu_id][stream_id], stream[gpu_id][stream_id]));
       kernel_callbacks[kernel_idx]<<<gridD, blockD, smemS, this->stream[gpu_id][stream_id]>>>(
                                                                  out_vector_device[gpu_id][stream_id].data,
                                                                  in_vector_device[gpu_id][stream_id].data,
                                                                  params_device[gpu_id][stream_id]
                                                                  );
       CCHECK(cudaEventRecord(end_event[gpu_id][stream_id], stream[gpu_id][stream_id]));
       _end = elapsedTime() - _start;
       logInfo("Launch async stream : ptr: %x, gpu_id : %d, stream_id : %d, Time : %f\n",
                       this->stream[gpu_id][stream_id],gpu_id, stream_id, _end);  
    }
    CCHECK(cudaGetLastError());
    //CCHECK(cudaDeviceSynchronize());

    logInfo("Params : premod : %d, premul : %d,  midx : %d, In Length : %d, Out Length : %d, Stride : %d, Padding Idx : %d, As mont : %d\n",
        params[i].premod, params[i].premul, params[i].midx, params[i].in_length, params[i].out_length, params[i].stride, params[i].padding_idx, params[i].as_mont);
    logInfo("Params (FFT cont.) : fft_Nx : %d, N_fftx : %d, fft_Ny : %d, N_ffty : %d, forward : %d\n",
        params[i].fft_Nx, params[i].N_fftx, params[i].fft_Ny, params[i].N_ffty, params[i].forward);
    logInfo("Kernel IDX :%d <<<%d, %d, %d>>> Time Elapsed Kernel : %f.sec, Return val : %d, In offset : %d, return offset : %d\n", 
          kernel_idx, gridD, blockD, smemS,end_kernel, config[i].return_val, config[i].in_offset, config[i].return_offset);
  }

    // retrieve kernel output data from GPU to host
  start_copy_out = elapsedTime();
  _start = elapsedTime();
  if (config[0].return_val){
     if (stream_id == 0){
        CCHECK(cudaMemcpy(
            out_vector_host->data,
            out_vector_device[gpu_id][stream_id].data + config[0].return_offset,
            out_vector_host->size,
            cudaMemcpyDeviceToHost));
     } else {
        //out_data_host[gpu_id][stream_id] = out_vector_host->data;
        out_data_host[gpu_id][stream_id] = out_vector_host->data;
        CCHECK(cudaMemcpyAsync(
            out_data_host[gpu_id][stream_id],
            out_vector_device[gpu_id][stream_id].data + config[0].return_offset,
            out_vector_host->size,
            cudaMemcpyDeviceToHost,
            stream[gpu_id][stream_id]));
        _end = elapsedTime() - _start;
        logInfo("Out Data : Ptr : %x, gpu_id : %u, stream_id : %u, Time : %f\n",
           out_data_host[gpu_id][stream_id], gpu_id, stream_id, _end);
     }
  }
 
  end_copy_out = elapsedTime() - start_copy_out;
  _total_end = elapsedTime() - _total_start;
  logInfo("Kernel total time : %f,\n\n", _total_end);

  logInfo("----- Info -------\n");
  logInfo("IVHS : %d, IVHL : %d, IVDS : %d, IVDL : %d\n",in_vector_host->size, 
		                                        in_vector_host->length,
						       	in_vector_device[gpu_id][stream_id].size,
						       	in_vector_device[gpu_id][stream_id].length);

  logInfo("OVHS : %d, OVHL : %d, OVDS : %d, OVDL : %d\n",out_vector_host->size,
		                                        out_vector_host->length, 
							out_vector_device[gpu_id][stream_id].size,
						       	out_vector_device[gpu_id][stream_id].length);

  logInfo("Time Elapsed Xfering in %d bytes : %f sec\n",
          in_vector_host->size, end_copy_in);
  logInfo("Time Elapsed Xfering out %d bytes : %f sec\n",
          out_vector_host->size, end_copy_out);

  return total_kernel;
}

void CUSnarks::streamDel(uint32_t gpu_id, uint32_t stream_id)
{
  uint32_t i;
  logInfo("Sync del : In Data ptr : %x, Out Data ptr : %x, In Params ptr : %x, gpu_id : %d, stream_id : %d\n\n\n",
      in_data_host[gpu_id][stream_id], out_data_host[gpu_id][stream_id], params_host[gpu_id][stream_id],
      gpu_id, stream_id);
  // free params, in_host_data and out_host_data

  // free params, in_host_data and out_host_data
  CCHECK(cudaFreeHost(in_data_host[gpu_id][stream_id]));
  CCHECK(cudaFreeHost(out_data_host[gpu_id][stream_id]));
  //TODO pending release params host
  //CCHECK(cudaFreeHost(params_host[gpu_id][stream_id]));

}

double CUSnarks::streamSync(uint32_t gpu_id, uint32_t stream_id)
{
  float kernel_time;

  logInfo("Sync Stream : Ptr : %x, gpu_id : %u, stream_id : %u\n",
           stream[gpu_id][stream_id], gpu_id, stream_id);
  logInfo("Sync Event  : Ptr : %x, gpu_id : %u, stream_id : %u\n",
           end_event[gpu_id][stream_id], gpu_id, stream_id);
  CCHECK(cudaStreamSynchronize(stream[gpu_id][stream_id]));
  //CCHECK(cudaEventSynchronize(end_event[gpu_id][stream_id]));

  /*
  CCHECK(cudaEventElapsedTime(
                &kernel_time,
                start_event[gpu_id][stream_id],
                end_event[gpu_id][stream_id]));
  */

  return  (double) kernel_time;
   
}

uint32_t * CUSnarks::streamGetOutputData(uint32_t gpu_id, uint32_t stream_id)
{   
   logInfo("Sync Out Data : Ptr :%x, gpu_id : %d, stream_id : %d \n", 
       out_data_host[gpu_id][stream_id], gpu_id, stream_id);

   return out_data_host[gpu_id][stream_id];
}

uint32_t  CUSnarks::streamGetOutputDataLen(uint32_t gpu_id, uint32_t stream_id)
{  
   logInfo("Sync Params : Ptr :%x, gpu_id : %d, stream_id : %d, len : %d\n", 
       params_host[gpu_id][stream_id], gpu_id, stream_id, 
       params_host[gpu_id][stream_id][0].out_length);
 
       
   return params_host[gpu_id][stream_id][0].out_length;
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
