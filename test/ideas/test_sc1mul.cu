#include <stdio.h>
#include "types.h"
#include "cusnarks_kernel.h"
#include "ecbn128.h"

#define N  (1024)
main()
{
    ECBN128 *ecbn128 = new ECBN128(N,123);
    
    uint32_t in_samples[N * NWORDS_256BIT * 3];
    uint32_t out_samples[N * NWORDS_256BIT * 3];
    vector_t in_vector, out_vector;
    double t;

    uint32_t p[] = { 4026531841, 1138881939, 2042196113,  674490440, 2172737629, 3092268470, 3778125865,  811880050};
    ecbn128->randu256(in_samples, N*6, p);
    uint32_t i;
    kernel_config_t kconfig;
    kernel_params_t kparams;

    in_vector.data = in_samples;
    in_vector.length = N * 3;
    out_vector.data = out_samples;
    out_vector.length = N*3;

    kconfig.blockD = 256;
    kconfig.gridD = 0;
    kconfig.smemS = 0;
    kconfig.kernel_idx = CB_EC_JAC_MUL1;

    kparams.stride = 1;
    kparams.premul = 0;
    kparams.premod = 0;
    kparams.midx   = MOD_FP;
    kparams.in_length = N*3;
    kparams.out_length = (N-2)*3;
    kparams.padding_idx = 0;

    t = ecbn128->kernelLaunch(&out_vector, &in_vector, &kconfig, &kparams,0,0,1);

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
