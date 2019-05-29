#include <stdio.h>
#include "types.h"
#include "utils_host.h"
#include "cusnarks_kernel.h"
#include "zpoly.h"
#include "ecbn128.h"

#define FFT_N  (5)
#define M (10)
#define N  (1024*1024)
#define NROOTS_1M (20)

static char roots_1M_filename[]="../../data/zpoly_roots_1M.bin";

main()
{
    ZCUPoly *z1poly = new ZCUPoly(2097154, 123);
    
    uint32_t *in_samples = (uint32_t *) malloc((2*N+2) * NWORDS_256BIT * sizeof(uint32_t));
    uint32_t *out_samples = (uint32_t *) malloc((N) * NWORDS_256BIT * sizeof(uint32_t));
    vector_t in_vector, out_vector;
    double t;

    uint32_t p[] = { 4026531841, 1138881939, 2042196113,  674490440, 2172737629, 3092268470, 3778125865,  811880050};


    z1poly->randu256(in_samples, N, p);
    readU256DataFile_h(&in_samples[N*NWORDS_256BIT],roots_1M_filename,1<<NROOTS_1M,1<<NROOTS_1M);
    uint32_t i;
    kernel_config_t kconfig[1];
    kernel_params_t kparams[1];

    in_vector.data = in_samples;
    in_vector.length = 2*N+2;
    out_vector.data = out_samples;
    out_vector.length = N;

    for (i=0; i<1; i++){
      kparams[i].in_length = 2*N + 2;
      kparams[i].out_length = N;
      kparams[i].stride = 2;
      kparams[i].premod = 0;
      kparams[i].midx   = MOD_FIELD;
      kparams[i].fft_Nx = (fft_size_t)FFT_N;
      kparams[i].fft_Ny = (fft_size_t)FFT_N;
      kparams[i].N_fftx =  M;
      kparams[i].N_ffty =  M;
      kparams[i].forward = 1;
      kparams[i].as_mont = 1;
      kconfig[i].smemS = 0;
     
      kconfig[i].blockD = 256;
      kconfig[i].gridD = 4096;
      kconfig[i].kernel_idx = CB_ZPOLY_FFT3DXX;
    }
    
    t = z1poly->kernelLaunch(&out_vector, &in_vector, kconfig, kparams,1);

    printf("Time : %f\n",t);
    printf("IN\n");
    for (i=0;i < 10; i++){
      printf("%u ",in_samples[i]);
    }
    printf("\n");
    printf("OUT\n");
    for (i=0;i < 10; i++){
      printf("%u ",out_samples[i]);
    }
    printf("\n");
}
