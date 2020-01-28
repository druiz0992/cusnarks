#include <stdio.h>
#include "types.h"
#include "cusnarks_kernel.h"
#include "utils_host.h"
#include "constants.h"
#include "u256.h"

#define N  (1024)
main()
{
    U256 *u256 = new U256(N, 123);
    
    uint32_t in_samples[N * NWORDS_256BIT];
    uint32_t out_samples[N/2 * NWORDS_256BIT];
    vector_t in_vector, out_vector;
    double t;
    uint32_t midx = MOD_FIELD;

    const uint32_t *p = CusnarksPGet((mod_t)midx);
    u256->randu256(in_samples, N, (uint32_t *)p);
    uint32_t i, n_kernels = 1;
    kernel_config_t kconfig;
    kernel_params_t kparams;

    printf("IN\n");
    for (i=0;i < 10; i++){
      printU256Number(&in_samples[i*NWORDS_256BIT]);
    }

    printf("\n");
    in_vector.data = in_samples;
    in_vector.length = N;
    out_vector.data = out_samples;
    out_vector.length = N/2;

    kconfig.blockD = 256;
    kconfig.gridD = 0;
    kconfig.smemS = 0;
    kconfig.return_val = 1;
    kconfig.kernel_idx = CB_U256_MULM;

    kparams.stride = 2;
    kparams.premod = 0;
    kparams.midx   = (mod_t) midx;
    kparams.in_length = N;
    kparams.out_length = N/2;
    kparams.padding_idx = 0;

    printf("Launching kernel\n");
    t = u256->kernelLaunch(&out_vector, &in_vector, &kconfig, &kparams, 0, 0,n_kernels);

    printf("Time : %f\n",t);
    printf("OUT\n");
    for (i=0;i < 10; i++){
      printU256Number(&out_samples[i*NWORDS_256BIT]);
    }
    printf("\n");
    
}
