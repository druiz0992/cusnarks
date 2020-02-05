#include <stdio.h>
#include <time.h>
#include "types.h"
#include "cusnarks_kernel.h"
#include "utils_host.h"
#include "constants.h"
#include "u256.h"

#define N  (4096)
main()
{
    U256 *u256 = new U256(N, time(NULL));
    
    uint32_t in_samples[N * NWORDS_256BIT];
    uint32_t out_samples1[N/2 * NWORDS_256BIT];
    uint32_t out_samples2[N/2 * NWORDS_256BIT];
    vector_t in_vector, out_vector;
    double t;
    uint32_t midx = MOD_FIELD;
    uint32_t done = 0;
    uint32_t n_errors = 0;
    uint32_t max_samples = 1<< 26;
    uint32_t n_samples = 0;

    const uint32_t *p = CusnarksPGet((mod_t)midx);
    uint32_t i, n_kernels = 1;
    kernel_config_t kconfig;
    kernel_params_t kparams;

    in_vector.data = in_samples;
    in_vector.length = N;
    out_vector.data = out_samples1;
    out_vector.length = N/2;

    kconfig.blockD = 256;
    kconfig.gridD = 0;
    kconfig.smemS = 0;
    kconfig.return_val = 1;
    kconfig.kernel_idx = CB_U256_MULM2;

    kparams.stride = 4;
    kparams.premod = 0;
    kparams.midx   = (mod_t) midx;
    kparams.in_length = N;
    kparams.out_length = N/2;
    kparams.padding_idx = 0;

    while(done == 0){
      u256->randu256(in_samples, N, (uint32_t *)p);

      // PTX
      out_vector.data = out_samples1;
      t = u256->kernelLaunch(&out_vector, &in_vector, &kconfig, &kparams, 0, 0,n_kernels);

      n_samples += N/2;
      if (n_samples >= max_samples){
        done = 1;
      }
    }
    
}
