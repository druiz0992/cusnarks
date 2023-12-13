#include <stdio.h>
#include "types.h"
#include "rng.h"
#include "cusnarks_kernel.h"
#include "ecbn128.h"

#define N  (1024)
main()
{
    ECBN128 *ecbn128 = new ECBN128(N,123);
    
    uint32_t in_samples[N * NWORDS_256BIT * 5];
    uint32_t out_samples[N * NWORDS_256BIT * 6];
    vector_t in_vector, out_vector;
    double t;

    uint32_t p[] = { 4026531841, 1138881939, 2042196113,  674490440, 2172737629, 3092268470, 3778125865,  811880050};
    ecbn128->randu256(in_samples, N*6, p);
    uint32_t i;
    kernel_config_t kconfig[2];
    kernel_params_t kparams[2];

    in_vector.data = in_samples;
    in_vector.length = N * 3;
    out_vector.data = out_samples;
    out_vector.length = N*3;

    for (i=0; i<2; i++){
      kconfig[i].blockD = 256;
      kconfig[i].gridD = 0;
      kconfig[i].smemS = kconfig[i].blockD/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4;
      kconfig[i].kernel_idx = CB_EC_JAC_MAD_SHFL;

      kparams[i].stride = ECP_JAC_OUTDIMS;
      kparams[i].premul = 1;
      kparams[i].premod = 0;
      kparams[i].midx   = MOD_FR;
      kparams[i].in_length = N * 3;
      kparams[i].out_length = 1 * ECP_JAC_OUTDIMS; 
      kparams[i].padding_idx = 0;

    }
    kconfig[1].blockD = 32;
    kconfig[1].gridD = 1;
    kparams[1].premul = 0;
    kparams[1].in_length = 12; 
    
    t = ecbn128->kernelLaunch(&out_vector, &in_vector, kconfig, kparams,2);

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
