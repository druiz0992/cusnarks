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
// File name  : utils_host.h
//
// Date       : 06/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of small utils functions for host
// ------------------------------------------------------------------

*/
#ifndef _UTILS_HOST_H_
#define _UTILS_HOST_H_


void utils_init_h(void);
void utils_free_h(void);
uint32_t utils_isinit_h(void);
uint32_t get_nprocs_h();
uint32_t launch_client_h( void * (*f_ptr) (void* ), pthread_t *workers, void *w_args, uint32_t size, uint32_t max_threads, uint32_t detach);
void wait_h(uint32_t thread_id, void (*f_ptr) (void *), void * args);
void fail_h();
int shared_new_h(void **shmem, unsigned long long size);
void shared_free_h(void *shmem, int shmid);
void init_barrier_h(uint32_t nthreads);
void del_barrier_h();

#endif
