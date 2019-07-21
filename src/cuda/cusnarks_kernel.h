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
// Date       : 12/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of cusnarks CUDA resources management
// ------------------------------------------------------------------

*/
#ifndef _CUSNARKS_H_
#define _CUSNARKS_H_

class CUSnarks {
    protected:
        vector_t in_vector_device;      // kernel input vector (device size)
        vector_t out_vector_device;     // kernel output vector (device side)
        kernel_params_t *params_device; // kernel params (device side)

        kernel_cb *kernel_callbacks;    // pointer to kernel callbacks
        //_RNG *rng;

        CUSnarks(uint32_t in_len, uint32_t in_size, 
		 uint32_t out_len, uint32_t out_size, kernel_cb *kcb);
        CUSnarks(uint32_t in_len, uint32_t in_size, 
		uint32_t out_len, uint32_t out_size, kernel_cb *kcb,
	      	uint32_t seed);
        ~CUSnarks();

        void allocateCudaResources(uint32_t in_size, uint32_t out_size);
        void allocateCudaCteResources(void);
        //void initRNG(uint32_t seed);
        double elapsedTime(void);

    public:

        static uint32_t init_constants;
        void rand(uint32_t *samples, uint32_t n_samples);
        void randu256(uint32_t *samples, uint32_t n_samples, uint32_t *mod);
        void saveFile(uint32_t *samples, uint32_t n_samples, char *fname);
        double kernelLaunch(
		vector_t *out_vector_host,
	       	vector_t *in_vector_host,
                kernel_config_t *config,
                kernel_params_t *params,
                uint32_t n_kernel);
        void getDeviceInfo();
};

#endif
