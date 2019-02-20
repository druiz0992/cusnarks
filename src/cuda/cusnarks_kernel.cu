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
#include "cuda.h"
#include "log.h"
#include "rng.h"
#include "cusnarks_kernel.h"


using namespace std;


// Prime information for finitie fields. Includes 3 numbers : p. p_ and r_ that 
// follow p x p_ - r * r_ = 1 whrere r is 1^256. This is used for Montgomery reduction
//
// There are two different set of primes (MOD_N)
__constant__ mod_info_t mod_info_ct[MOD_N];


// Group
//  p1 = 21888242871839275222246405745257275088696311157297823662689037894645226208583L
// 'Pp': 111032442853175714102588374283752698368366046808579839647964533820976443843465L,
// 'R': 115792089237316195423570985008687907853269984665640564039457584007913129639936L,
// 'R3modP': 14921786541159648185948152738563080959093619838510245177710943249661917737183L,
// 'Rbitlen': 256,
// 'Rmask': 115792089237316195423570985008687907853269984665640564039457584007913129639935L,
// 'RmodP': 6350874878119819312338956282401532409788428879151445726012394534686998597021L,
// 'Rp': 20988524275117001072002809824448087578619730785600314334253784976379291040311


// Field
//p2 = 21888242871839275222246405745257275088548364400416034343698204186575808495617L
// 'Pp': 52454480824480482120356829342366457550537710351690908576382634413609933864959L,
// 'R': 115792089237316195423570985008687907853269984665640564039457584007913129639936L,
// 'R3modP': 5866548545943845227489894872040244720403868105578784105281690076696998248512L,
// 'Rbitlen': 256,
// 'Rmask': 115792089237316195423570985008687907853269984665640564039457584007913129639935L,
// 'RmodP': 6350874878119819312338956282401532410528162663560392320966563075034087161851L,
// 'Rp': 9915499612839321149637521777990102151350674507940716049588462388200839649614L}

static uint32_t mod_info_init[] = {
        3632069959, 1008765974, 1752287885, 2541841041, 2172737629, 3092268470, 3778125865,  811880050, // p_group
        3834012553, 2278688642,  516582089, 2665381221,  406051456, 3635399632, 2441645163, 4118422199, // pp_group
        21690935,   3984885834,   41479672, 3944751749, 3074724569, 3479431631, 1508230713,  778507633, // rp_group

        4026531841, 1138881939, 2042196113,  674490440, 2172737629, 3092268470, 3778125865,  811880050, // p_field
        4026531839, 3269588371, 1281954227, 1703315019, 2567316369, 3818559528,  226705842, 1945644829, // pp_field
        1840322894, 3696992261, 3776048263,  151975337, 2931318109, 3357937124, 2193970460,   367786321 // rp_field
};



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


// Group prime
// b = 19052624634359457937016868847204597229365286637454337178037183604060995791063L
// Gx = 6350874878119819312338956282401532409788428879151445726012394534686998597021L
// Gy = 12701749756239638624677912564803064819576857758302891452024789069373997194042L
// G2x[0] = 11461925177900819176832270005713103520318409907105193817603008068482420711462L
// G2x[1] = 18540402224736191443939503902445128293982106376239432540843647066670759668214L
// G2y[0] = 9496696083199853777875401760424613833161720860855390556979200160215841136960L
// G2y[1] = 6170940445994484564222204938066213705353407449799250191249554538140978927342L

// Field prime
// b = 19052624634359457937016868847204597231584487990681176962899689225102261485553L
// Gx = 6350874878119819312338956282401532410528162663560392320966563075034087161851L
// Gy = 12701749756239638624677912564803064821056325327120784641933126150068174323702L
// G2x[0] = 3440318644824060289325407041038137137632482455953552081609447686580196514077L
// G2x[1] = 15555376658169732961166172612384867299105908138835914639331977638675822381717L
// G2y[0] = 1734729704421626988316384622007148076088981578411341419187802115428207738199L
// G2y[1] = 7947406416180328355183476715314321281876370016171195924215639649670494139363L 

static uint32_t ecbn128_params_init [] = {
    1353525463, 2048379561, 3780452793,  527090042, 1768673924,  860613198, 3457654158,  706701124,   // b_group
    3314486685, 3546104717, 4123462461,  175696680, 2021213740, 1718526831, 2584207151,  235567041,   // Gx_group
    2334006074, 2797242139, 3951957627,  351393361, 4042427480, 3437053662,  873447006,  471134083,   // Gy group
      45883430, 2390996433, 1232798066, 3706394933, 2541820639, 4223149639, 2945863739,  425146433,   // G2x[0] group
    2288773622, 1637743261, 4120812408, 4269789847,  589004286, 4288551522, 2929607174,  687701739,   // G2x[1] group
    2823577920, 2947838845, 1476581572, 1615060314, 1386229638,  166285564,  988445547,  352252035,   // G2y[0] group
    3340261102, 1678334806,  847068347, 3696752930,  859115638, 1442395582, 2482857090,  228892902,   // G2y[1] group

    4026531825,   96640084, 3726796669, 2767545280, 1768673930,  860613198, 3457654158,  706701124,  // b field
    1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,  // Gx field
    2684354550, 1496082488, 1052875347, 1845030187, 4042427484, 3437053662,  873447006,  471134083,  // Gy field
    3570833693,  985424601, 3020216734, 2567113431,  703417746, 1422701227, 3337448090,  127608510,  // G2x[0] field
    1354730133, 2060109890, 3374016652, 3251713708,  786468672, 1666612222, 3296074718,  576980987,  // G2x[1] field
    2182790487,  762510808, 2006819228, 3200553925, 2281110735, 3404365023, 3840597178,   64344700,  // G2y[0] field
     939242467, 1534311190, 2907306748, 1573550191,  646343074, 2690260169, 2616010917,  294785687   // G2y[1] field
};


// Additional constants
__constant__ misc_const_t misc_const_ct[MOD_N];

// group
// 1 => 6350874878119819312338956282401532409788428879151445726012394534686998597021L
// 2 => 12701749756239638624677912564803064819576857758302891452024789069373997194042L
// 3 => 19052624634359457937016868847204597229365286637454337178037183604060995791063L
// 4 => 3515256640640002027109419384348854550457404359307959241360540244102768179501L
// 8 => 7030513281280004054218838768697709100914808718615918482721080488205536359002L
// 12 (4b) => 10545769921920006081328258153046563651372213077923877724081620732308304538503L
// 24 (8b)=> 21091539843840012162656516306093127302744426155847755448163241464616609077006L
// field
// 1 => 6350874878119819312338956282401532410528162663560392320966563075034087161851L
// 2 => 12701749756239638624677912564803064821056325327120784641933126150068174323702L
// 3 => 19052624634359457937016868847204597231584487990681176962899689225102261485553L
// 4 =>  3515256640640002027109419384348854553564286253825534940168048113560540151787L
// 8 => 7030513281280004054218838768697709107128572507651069880336096227121080303574L
// 12 (4b) => 10545769921920006081328258153046563660692858761476604820504144340681620455361L
// 24 (8b) => 21091539843840012162656516306093127321385717522953209641008288681363240910722L

static uint32_t misc_const_init[] = {
         3314486685, 3546104717, 4123462461,  175696680, 2021213740, 1718526831, 2584207151,  235567041,    // 1 group
         2334006074, 2797242139, 3951957627,  351393361, 4042427480, 3437053662,  873447006,  471134083,    // 2 group
         1353525463, 2048379561, 3780452793,  527090042, 1768673924,  860613198, 3457654158,  706701124,   // 3_group
         1035942189,  290751008, 1856660074, 2455912978, 1617150034, 3781838855, 2263735443,  130388115,    // 4 group
         2071884378,  581502016, 3713320148,  616858660, 3234300069, 3268710414,  232503591,  260776231,   // 8 group
         3107826567,  872253024, 1275012926, 3072771639,  556482807, 2755581974, 2496239035,  391164346,    // 4b group
	 1920685838, 1744506049, 2550025852, 1850575982, 1112965615, 1216196652,  697510775,  782328693,    // 8b group 
         3314486685, 3546104717, 4123462461,  175696680, 2021213740, 1718526831, 2584207151,  235567041,    // inf group
                  0,          0,          0,          0,          0,          0,          0,          0,
         3314486685, 3546104717, 4123462461,  175696680, 2021213740, 1718526831, 2584207151,  235567041,    // 

         1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,   // 1 field
         2684354550, 1496082488, 1052875347, 1845030187, 4042427484, 3437053662,  873447006,  471134083,   // 2 field
         4026531825,   96640084, 3726796669, 2767545280, 1768673930,  860613198, 3457654158,  706701124,  // 3 field
         1342177259, 1853283037,   63554581, 3015569934, 1617150043, 3781838855, 2263735443,  130388115,    // 4 field
         2684354518, 3706566074,  127109162, 1736172572, 3234300087, 3268710414,  232503591,  260776231,  // 8 field
         4026531777, 1264881815,  190663744,  456775210,  556482835, 2755581974, 2496239035,  391164346,    // 4b field
	 3758096258, 2529763631,  381327488,  913550420, 1112965670, 1216196652,  697510775,  782328693,     // 8b field
         1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,   // inf
                  0,          0,          0,          0,          0,          0,          0,          0,    
         1342177275, 2895524892, 2673921321,  922515093, 2021213742, 1718526831, 2584207151,  235567041,   // 
};

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
  initRNG(seed);
}

/*
   Reserve GPU memory for input and output vectors and input kernel params (global memory),
   as well as some constant info (constant memory)
 */
void CUSnarks::allocateCudaResources(uint32_t in_size, uint32_t out_size)
{
  mod_info_t mod_h[MOD_N];
  ecbn128_t ecbn128_h[MOD_N];
  misc_const_t misc_h[MOD_N];

  // Allocate kernel input and putput data vectors in global memory 
  CCHECK(cudaMalloc((void**) &this->in_vector_device.data, in_size));
  CCHECK(cudaMalloc((void**) &this->out_vector_device.data, out_size));

  // Allocate kernel params in global memory 
  CCHECK(cudaMalloc((void**) &this->params_device, sizeof(kernel_params_t)));

  // constants ->  Initialize data and copy to constant memory
  memcpy(mod_h,     mod_info_init,           sizeof(mod_info_t)    * MOD_N); // prime info
  memcpy(ecbn128_h, ecbn128_params_init,     sizeof(ecbn128_t)     * MOD_N); // ecbn128
  memcpy(misc_h,    misc_const_init,         sizeof(misc_const_t)  * MOD_N); // misc

  // Copy modulo info to device constant
  CCHECK(cudaMemcpyToSymbol(mod_info_ct,       mod_h,     MOD_N * sizeof(mod_info_t)));  // prime info
  CCHECK(cudaMemcpyToSymbol(ecbn128_params_ct, ecbn128_h, MOD_N * sizeof(ecbn128_t)));   // ecbn128
  CCHECK(cudaMemcpyToSymbol(misc_const_ct,    misc_h,    MOD_N * sizeof(misc_const_t)));// misc
}

/*
   Initialize PCG random number generator  
   http://www.pcg-random.org/

   IF seed is 0, random seed is taken from urand.

   NOTE : when seed is 0, generator breaks
   TODO :  Fix seed = 0
*/
void CUSnarks::initRNG(uint32_t seed)
{
  if (seed == 0){ rng =  _RNG::get_instance(); }
  else { rng = _RNG::get_instance(seed); }
}
/*
   Generate N 32 bit random samples
*/
void CUSnarks::rand(uint32_t *samples, uint32_t n_samples)
{
    uint32_t size_sample = in_vector_device.size / (in_vector_device.length * sizeof(uint32_t));
    rng->randu32(samples, n_samples * size_sample);
}

/*
   Free memory allocated in GPU:
*/
CUSnarks::~CUSnarks()
{
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
void CUSnarks::kernelLaunch(
                uint32_t kernel_idx,
		vector_t *out_vector_host,
	       	vector_t *in_vector_host,
                kernel_config_t *config,
                kernel_params_t *params)
{
  logInfo("IVHS : %d, IVHL : %d, IVDS : %d, IVDL : %d\n",in_vector_host->size, 
		                                        in_vector_host->length,
						       	in_vector_device.size,
						       	in_vector_device.length);

  logInfo("OVHS : %d, OVHL : %d, OVDS : %d, OVDL : %d\n",out_vector_host->size,
		                                        out_vector_host->length, 
							out_vector_device.size,
						       	out_vector_device.length);

  // check input lengths do not exceed reserved amount
  if (in_vector_host->length > in_vector_device.length) { return; }
  if (out_vector_host->length > out_vector_device.length) { return; }

  in_vector_host->size = in_vector_host->length * (in_vector_device.size / in_vector_device.length  );
  out_vector_host->size = out_vector_host->length * (out_vector_device.size / out_vector_device.length );

  double start, end_copy_in, end_kernel, end_copy_out;
  int blockD, gridD;

  logInfo("IVHS : %d, IVHL : %d, IVDS : %d, IVDL : %d\n",in_vector_host->size, 
		                                        in_vector_host->length,
						       	in_vector_device.size,
						       	in_vector_device.length);

  logInfo("OVHS : %d, OVHL : %d, OVDS : %d, OVDL : %d\n",out_vector_host->size,
		                                        out_vector_host->length, 
							out_vector_device.size,
						       	out_vector_device.length);

  // measure data xfer time Host -> Device
  start = elapsedTime();
  CCHECK(cudaMemcpy(in_vector_device.data, in_vector_host->data, in_vector_host->size, cudaMemcpyHostToDevice));
  CCHECK(cudaMemcpy(params_device, params, sizeof(kernel_params_t), cudaMemcpyHostToDevice));
  end_copy_in = elapsedTime() - start;
 
  // configure kernel. Input parameter invludes block size. Grid is calculated 
  // depending on input data length and stride (how many samples of input data are 
  // used per thread
  blockD = config->blockD;
  if (config->gridD == 0){
     config->gridD = (blockD + in_vector_host->length/params->stride - 1) / blockD;
  }
  gridD = config->gridD;


  // launch kernel
  start = elapsedTime();
  kernel_callbacks[kernel_idx]<<<gridD, blockD, config->smemS>>>(out_vector_device.data, in_vector_device.data, params_device);
  CCHECK(cudaGetLastError());
  CCHECK(cudaDeviceSynchronize());
  end_kernel = elapsedTime() - start;

  // retrieve kernel output data from GPU to host
  start = elapsedTime();
  CCHECK(cudaMemcpy(out_vector_host->data, out_vector_device.data, out_vector_host->size, cudaMemcpyDeviceToHost));
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

  logInfo("Params : premod : %d, midx : %d, In Length : %d, Out Length : %d\n",params->premod, params->midx, params->in_length, params->out_length);
  logInfo("Kernel IDX :%d <<<%d, %d, %d>>> Time Elapsed Kernel : %f.sec\n", 
          kernel_idx, gridD, blockD, config->smemS,end_kernel);
  logInfo("Time Elapsed Xfering in %d bytes : %f sec\n",
          in_vector_host->size, end_copy_in);
  logInfo("Time Elapsed Xfering out %d bytes : %f sec\n",
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
