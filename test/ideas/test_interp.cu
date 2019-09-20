#include <stdio.h>
#include "types.h"
#include "utils_host.h"
#include "cusnarks_kernel.h"
#include "zpoly.h"

#define NROOTS_1M (20)

static char roots_1M_filename[]="../../data/zpoly_roots_1M.bin";


static uint32_t blockD[] = {256, 256, 256, 256, 256, 256, 256,
                            256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256};

static uint32_t gridD[] = {512, 512, 512, 512, 512, 512, 512, 512, 512,
                            512, 512, 512, 512, 512, 512, 512, 512, 512};
 

//static uint32_t kernel_idx[] = {13, 14, 15, 16, 13, 14, 15, 16, 17,
                              //13, 14, 15, 16, 13, 14, 15, 16, 17};
static uint32_t kernel_idx[] = {13, 14, 15, 16, 17, 14, 15, 16, 17};
 
//static uint32_t return_offset[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static uint32_t return_val[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static uint32_t smemS[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


//static uint32_t N_fftx[] = {4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5};
//static uint32_t N_ffty[] =  {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
static uint32_t N_fftx[] = {9,9,9,9,9,9,9,9,9,9};
static uint32_t N_ffty[] =  {8,8,8,8,8,8,8,8,8,8};
static uint32_t as_mont[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//static uint32_t fft_Nx[] =  {2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0};
//static uint32_t fft_Ny[] =  {2, 2, 2, 2, 3, 3, 2, 2, 0, 2, 2, 2, 2, 3, 3, 2, 2, 0};
static uint32_t fft_Nx[] =  {4,4,4,4,4, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0};
static uint32_t fft_Ny[] =  {2, 2, 2, 2, 3, 3, 2, 2, 0, 2, 2, 2, 2, 3, 3, 2, 2, 0};
static uint32_t forward[] =  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static uint32_t in_length[] =  {525058, 131072, 131072, 131072, 131072, 131072, 131072, 131072,
                          131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072};
static uint32_t midx[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static uint32_t out_length = 262144;
static uint32_t padding_idx[] = {131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072,
  131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072};
static uint32_t premod[] =  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static uint32_t premul[] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0};
static uint32_t stride[] = {768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768,
  768, 768, 768, 768, 768, 768}; 

#define NSAMPLES (1<<17)
#define NROOTS (1<<20)
#define NSAMPLES_P1 (1<<8)
#define NSAMPLES_P2 (1<<9)


static uint32_t scalerExt[] = { 4113958913, 2342081442, 2241147828, 2080724014, 1587943324,
                                3896207758, 3120050064,  811873856};
static uint32_t scalerMont[] = {   0,     0,     0,     0,     0,     0,     0, 32768};

main()
{
    uint32_t *samples = (uint32_t *)malloc(NSAMPLES*sizeof(uint32_t)*NWORDS_256BIT);
    uint32_t *roots_1M = (uint32_t *)malloc(NROOTS*sizeof(uint32_t)*NWORDS_256BIT);
    uint32_t *roots_4D = (uint32_t *) malloc(2*NSAMPLES*sizeof(uint32_t)*NWORDS_256BIT);
    uint32_t *roots_3D = (uint32_t *) malloc(NSAMPLES*sizeof(uint32_t)*NWORDS_256BIT);
    uint32_t *roots_2D1 = (uint32_t *) malloc(NSAMPLES_P1*sizeof(uint32_t)*NWORDS_256BIT);
    uint32_t *roots_2D2 = (uint32_t *) malloc(NSAMPLES_P2*sizeof(uint32_t)*NWORDS_256BIT);

    uint32_t *in_samples = (uint32_t *)malloc((4*NSAMPLES+NSAMPLES_P1+NSAMPLES_P2+2)*sizeof(uint32_t) *NWORDS_256BIT);
    uint32_t *out_samples = (uint32_t *)malloc((2*NSAMPLES)*sizeof(uint32_t) *NWORDS_256BIT);

    vector_t in_vector, out_vector;
    double t;

    readU256DataFile_h(samples, "../c/aux_data/zpoly_samples_fft4d.bin",NSAMPLES, NSAMPLES);
    readU256DataFile_h(roots_1M,roots_1M_filename, NROOTS, NROOTS);
    readU256DataFile_h(roots_3D,roots_1M_filename, NROOTS, NSAMPLES);
    readU256DataFile_h(roots_4D,roots_1M_filename, NROOTS, 2*NSAMPLES);
    readU256DataFile_h(roots_2D1,roots_1M_filename, NROOTS, NSAMPLES_P1);
    readU256DataFile_h(roots_2D2,roots_1M_filename, NROOTS, NSAMPLES_P2);


    memcpy(in_samples,samples,NSAMPLES*NWORDS_256BIT*sizeof(uint32_t));
    memcpy(&in_samples[NSAMPLES],samples,NSAMPLES*NWORDS_256BIT*sizeof(uint32_t));
    memcpy(&in_samples[2*NSAMPLES],roots_2D1,NSAMPLES_P1*NWORDS_256BIT*sizeof(uint32_t));
    memcpy(&in_samples[2*NSAMPLES+NSAMPLES_P1],roots_2D2,NSAMPLES_P2*NWORDS_256BIT*sizeof(uint32_t));
    memcpy(&in_samples[2*NSAMPLES+NSAMPLES_P1+NSAMPLES_P2],roots_3D,NSAMPLES*NWORDS_256BIT*sizeof(uint32_t));
    memcpy(&in_samples[3*NSAMPLES+NSAMPLES_P1+NSAMPLES_P2],roots_4D,NSAMPLES*NWORDS_256BIT*sizeof(uint32_t));
    memcpy(&in_samples[3*NSAMPLES+NSAMPLES_P1+NSAMPLES_P2+1],scalerExt,NWORDS_256BIT*sizeof(uint32_t));
    memcpy(&in_samples[3*NSAMPLES+NSAMPLES_P1+NSAMPLES_P2+2],scalerMont,NWORDS_256BIT*sizeof(uint32_t));

    kernel_config_t kconfig[18];
    kernel_params_t kparams[18];

    in_vector.data = in_samples;
    in_vector.length = 4*NSAMPLES+NSAMPLES_P1+NSAMPLES_P2+2;
    out_vector.data = out_samples;
    out_vector.length = 2*NSAMPLES;

    for (uint32_t i=0; i<1; i++){
      kconfig[i].gridD = gridD[i];
      kconfig[i].return_val  = return_val[i];
      kconfig[i].blockD = blockD[i];
      kconfig[i].kernel_idx = kernel_idx[i];
      kconfig[i].smemS = smemS[i];
      kconfig[i].return_offset = return_offset[i];


      kparams[i].stride = stride[i];
      kparams[i].forward = forward[i];
      kparams[i].in_length = in_length[i];
      kparams[i].N_ffty = N_ffty[i];
      kparams[i].N_fftx = N_fftx[i];
      kparams[i].as_mont = as_mont[i];
      kparams[i].padding_idx = padding_idx[i];
      kparams[i].premul = premul[i];
      kparams[i].fft_Nx = (fft_size_t)fft_Nx[i];
      kparams[i].fft_Ny = (fft_size_t)fft_Ny[i];
      kparams[i].out_length = out_length;
      kparams[i].premod = premod[i];
      kparams[i].midx = (mod_t) midx[i];
    }
    
    ZCUPoly *z1poly = new ZCUPoly(5 * NSAMPLES, 123);
    t = z1poly->kernelLaunch(&out_vector, &in_vector, kconfig, kparams,1);

    printf("Time : %f\n",t);
    printf("IN\n");
    for (uint32_t i=0;i < 10; i++){
      printf("%u ",in_samples[i]);
    }
    printf("\n");
    printf("OUT\n");
    for (uint32_t i=0;i < 10; i++){
      printf("%u ",out_samples[i]);
    }
    printf("\n");
}
