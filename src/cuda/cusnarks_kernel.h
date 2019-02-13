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
//  Definition of cusnarks CUDA resources
// ------------------------------------------------------------------

*/
#ifndef _CUSNARKS_H_
#define _CUSNARKS_H_

class CUSnarks {
    private:
        uint32_t *in_vector_device;      // pointer to device input buffer
        uint32_t *out_vector_device;     // pointer to device output buffer
	uint32_t *p;                     // prime number
        const uint32_t  in_vector_len;   // array len
        _RNG *rng;

        void copyVectorToDevice(const uint32_t *in_vector_host, uint32_t in_size);
        void copyVectorFromDevice(uint32_t *out_vector_host, uint32_t out_size);
        void allocateCudaResources(const uint32_t *p, uint32_t in_size, uint32_t out_size);
        void initRNG(uint32_t seed);

    public:

        CUSnarks(const uint32_t *p, uint32_t len, uint32_t in_size, uint32_t out_size);
        CUSnarks(const uint32_t *p, uint32_t len, uint32_t in_size, uint32_t out_size, uint32_t seed);
        ~CUSnarks();
        void CUSnarks::rand(uint32_t *samples, uint32_t n_samples, uint32_t size);
        template<typename kernelFunction, typename... kernelParameters>
            void CUSnarks::kernelLaunch(
		uint32_t *out_vector_host,
	       	const uint32_t *in_vector_host,
	        uint32_t in_size,
		uint32_t out_size,
		const kernelFunction& kernel_function,
		kernelParameters... kernel_extra_params);
};

#endif
