//
//  In-place transpose of rectangular matrices (For inclusion in Armadillo C++ linear algebra library)
//
//  Inspired by : Gustavson, F. G., & Swirszcz, T. (2007). In-place transposition of rectangular matrices.
//                In Applied Parallel Computing. State of the Art in Scientific Computing (pp. 560-569).
//                Springer Berlin Heidelberg.
//
//  Created by : Alexandre Drouin
//  Date : 2013-09-30
//
//  Make with : g++ itranspose_tests.cpp -o tests -O3 -lboost_unit_test_framework -larmadillo

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "C++ Unit Tests for in-place transposition of rectangular matrices"

#include <iostream>
#include <boost/test/unit_test.hpp>
#include <armadillo>
#include <time.h>


using namespace arma;
using namespace std;

template<class TYPE>
void itranspose(Mat<TYPE>& A)
{
    const unsigned int m = A.n_rows;
    const unsigned int n = A.n_cols;
    
    if (m == n) {
        for (int i = 0; i < m - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                TYPE tmp = A(j, i);
                A(j, i) = A(i, j);
                A(i, j) = tmp;
            }
        }
    }
    else
    {
        A.set_size(A.n_cols, A.n_rows);
        
        vector<bool> visited(A.n_cols * A.n_rows);
        unsigned int *r = (unsigned int *)malloc(A.n_cols * A.n_rows * sizeof(unsigned int));
        unsigned int *r1 = (unsigned int *)malloc(A.n_cols * A.n_rows * sizeof(unsigned int));
        unsigned int idx=0;
        unsigned int nel=0;
        for (int row = 0; row < n; row++) {
            for (unsigned int col = 0; col < m; col++) {
                
                unsigned int pos = col * n + row;
                
                if (!visited[pos]) {
                    unsigned int curr_pos = pos;
                    
                    TYPE val = A(row, col);
                    
                    r[idx++] = curr_pos;
                    //printf("%d ", curr_pos);
                    while (!visited[curr_pos]) {
                        r1[idx-1] = ++nel;
                        //printf("%d ", curr_pos);
                        visited[curr_pos] = true;
                        
                        unsigned int A_j = curr_pos / m;
                        unsigned int A_i = curr_pos - m * A_j;
                        
                        TYPE tmp = A(A_j, A_i);
                        A(A_j, A_i) = val;
                        val = tmp;
                        
                        curr_pos = A_i*n + A_j;
                    }
                    //printf("\n");
                } else {
                  nel=0;
                } 
            }
        }
        unsigned int step = r[2] - r[1], first_el=r[1], group=1, ncyc=0, nit=0;
        for(unsigned int col=1; col < idx-1; col++){
          if(r[col+1] - r[col] == step) {
             group++;
          } else {
            ncyc +=4;
            nit++;
            printf("%d, %d, %d, %d,", first_el, step, group, r1[col]);
            group=1;
            first_el = r[col+1];
            step = r[col+2] - r[col+1];
            if (nit % 4 == 0){
             printf("\n");
            }
          }
          
        }
        free(r);
        printf("\n%d\n",ncyc);
    }
}

template<class TYPE>
void itranspose2(Mat<TYPE>& A,const unsigned int m1, const unsigned int n1)
{
    const unsigned int m = A.n_rows;
    const unsigned int n = A.n_cols;

    const unsigned int nelems = m1 + n1;
    const unsigned int ncycles = (m * n - 2) / (nelems);

    unsigned int cur_pos = n;
    const unsigned int offset = 2*n;
    const unsigned int mask = (1 << (m1 + n1))-1;
    const unsigned int MN_1 = m * n - 1;
    
    A.set_size(A.n_cols, A.n_rows);
        
    for (int ccount=0; ccount < ncycles; ccount++){
       for(int elcount=0; elcount < nelems; elcount++){
          TYPE val = A(cur_pos%n, cur_pos/n);
          unsigned int tras_pos = m * cur_pos;          
          if (tras_pos > MN_1){
            tras_pos -= MN_1;
          }

          unsigned int A_j = tras_pos / m;
          unsigned int A_i = tras_pos % m;
                      
          TYPE tmp = A(A_j, A_i);
          A(A_j, A_i) = val;
          val = tmp;
                       
          cur_pos = tras_pos * m; 
          if (cur_pos > MN_1){
             cur_pos -= MN_1;
          }
        }
        cur_pos = (cur_pos + offset) % mask;
  } 
}


BOOST_AUTO_TEST_CASE( transpose_column_matrix )
{
    double start, end;

    unsigned int min_m = 10;
    unsigned int max_m = 11;
    unsigned int i;

    for(i=min_m; i< max_m; i++){
    
      unsigned int m1 = i; 
      unsigned int n1 = i+1;
      unsigned int m = 1<<m1;
      unsigned int n = 1<<n1;
    
      fmat A(m, n);
      A.randu();
      
      start = (double)clock()/CLOCKS_PER_SEC;
      fmat B = A.t();
      end = (double)clock()/CLOCKS_PER_SEC;
      printf("Arma : %f\n", end-start);
      
      printf("--- %d --- \n",i);
      start = (double)clock()/CLOCKS_PER_SEC;
      //itranspose2(A,m1, n1);
      itranspose(A);
      end = (double)clock()/CLOCKS_PER_SEC;
      printf("Ix : %f\n", end-start);
      
      for (unsigned int i = 0; i < n; i++) {
          for (unsigned int j = 0; j < m; j++) {
              BOOST_CHECK_CLOSE(B(i,j), A(i,j), 0.0001);
          }
      }
   }
}

#if 0

BOOST_AUTO_TEST_CASE( transpose_row_matrix )
{
    const unsigned int m = 300;
    const unsigned int n = 1000;
    
    fmat A(m, n);
    A.randu();
    
    fmat B = A.t();
    
    itranspose(A);
    
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) {
            BOOST_CHECK_CLOSE(B(i,j), A(i,j), 0.0001);
        }
    }
}

BOOST_AUTO_TEST_CASE( transpose_square_matrix )
{
    const unsigned int m = 333;
    const unsigned int n = 333;
    
    fmat A(m, n);
    A.randu();
    
    fmat B = A.t();
    
    itranspose(A);
    
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) {
            BOOST_CHECK_CLOSE(B(i,j), A(i,j), 0.0001);
        }
    }
}

#endif
