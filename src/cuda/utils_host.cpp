
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
*/

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : utils_host.cpp
//
// Date       : 6/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Util functions for host. 
//
// ------------------------------------------------------------------
#include <stdlib.h>
#include <cassert>
#include <sys/sysinfo.h>
#include <pthread.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "types.h"
#include "utils_host.h"

static  pthread_barrier_t utils_barrier;  
static  uint32_t utils_nprocs = 1;        
static  uint32_t utils_mproc_init = 0;  

//////

/*
  init
*/
void utils_init_h()
{
  #ifdef PARALLEL_EN
  if (utils_mproc_init) {
    return;
  }

  utils_nprocs = get_nprocs_h(); 
  omp_set_num_threads(utils_nprocs);
  utils_mproc_init = 1;

  #endif

}

uint32_t utils_isinit_h(void)
{
  return utils_mproc_init;
}

void init_barrier_h(uint32_t nthreads)
{
  if (pthread_barrier_init(&utils_barrier, NULL, nthreads) != 0){
     exit(1);
  }
}

/*
  Wait until all threads reach this point. If f_ptr is not null, thread 0 can execute a function
*/
void wait_h(uint32_t thread_id, void (*f_ptr) (void *), void * args)
{
  pthread_barrier_wait(&utils_barrier);
  if (f_ptr){
     if (thread_id == 0){
       f_ptr(args);
     }
     pthread_barrier_wait(&utils_barrier);
  }
}

/*
  Launch threads
*/
uint32_t launch_client_h( void * (*f_ptr) (void* ), pthread_t *workers, void *w_args, uint32_t size, uint32_t max_threads, uint32_t detach)
{
  uint32_t i;

  for (i=0; i < max_threads; i++)
  {
     //printf("Thread %d : start_idx : %d, last_idx : %d . ptr : %x\n", i, w_args[i].start_idx,w_args[i].last_idx, f_ptr);
     if ( pthread_create(&workers[i], NULL, f_ptr, (void *) w_args+i*size) ){
       return 0;
     }
    if (detach){
      pthread_detach(workers[i]);
    }
  }

  if (detach == 0){
    //printf("Max threads : %d\n",w_args[0].max_threads);
    for (i=0; i < max_threads; i++){
      pthread_join(workers[i], NULL);
    }
  }
  return 1;
}

/*
  Get number of cores
*/
uint32_t get_nprocs_h()
{
  uint32_t max_cores = get_nprocs_conf();

  return max_cores;
}

void del_barrier_h(void)
{
  pthread_barrier_destroy(&utils_barrier);
}

void utils_free_h()
{
  #ifdef PARALLEL_EN
  if (utils_mproc_init) {
     utils_nprocs = 1;
     utils_mproc_init=0;
  }
  #endif
}


/* 
  Create Shared Memory Buffer

  void **shmem            : pointer to address of shared memory we want to create
  unsgined long long size : size in bytes of shared memory
  
  Returns shared memory ID (>0), and negative number if KO

  See example in test/ideas/test_shmem_read.c and test/ideas/test_shmem_create.c
*/
int shared_new_h(void **shmem, unsigned long long size)
{
  int shmid;
  // give your shared memory an id, anything will do
  key_t key = SHMEM_WITNESS_KEY;
 
  // Setup shared memory
  if ((shmid = shmget(key, size, IPC_CREAT | 0666)) < 0)
  {
     return -1;
  }
  // Attached shared memory
  if ((*shmem = shmat(shmid, NULL, 0)) == (char *) -1)
  {
     return -1;
  }

  return shmid;
}
/*
  Destroy Shared Memory Buffer 
  
  See example in test/ideas/test_shmem_read.c and test/ideas/test_shmem_create.c
*/
void shared_free_h(void *shmem, int shmid)
{
   // Detach and remove shared memory
   shmdt(shmem);
   shmctl(shmid, IPC_RMID, NULL);
}
	
void fail_h(void)
{
  assert(NULL);
}


