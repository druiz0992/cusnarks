#include <stdio.h>
#include <x86intrin.h>
#include <time.h>
#include <string.h>
#include <assert.h>

#define MAX_32DIGIT (0xFFFFFFFFUL)
#define MAX_64DIGIT (0xFFFFFFFFFFFFFFFFULL)

#define NEG(X) (~(X))

#define NWORDS_256BIT (8)
#define NWORDS_256BIT_FIOS (11)

#define NTESTS (1<<25)

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

static uint32_t NPrime[] = {4026531839, 3269588371, 1281954227, 1703315019, 2567316369, 3818559528,  226705842, 1945644829 };
static uint32_t N[] = {      4026531841, 1138881939, 2042196113,  674490440, 2172737629, 3092268470, 3778125865,  811880050};

void init_u256(uint32_t *x);
void print_u256(char *s,  uint32_t *x);
int comp_u256(uint32_t *x, uint32_t *y);

uint32_t add_test1(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits);
uint32_t add_test2(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits);
uint32_t add_test3(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits);
uint32_t add_test4(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits);
uint32_t add_test5(uint32_t w[], uint32_t u[], uint32_t v[], size_t ndigits);
uint64_t add_test6(uint64_t *c, uint64_t *a, uint64_t *b);
void add_Test1(void);
void add_Test2(void);

void comp_Test1(void);
int32_t comp_test1(uint32_t *a, uint32_t *b, uint32_t ndigits);
int32_t comp_test2(uint32_t *a, uint32_t *b, uint32_t ndigits);
int32_t comp_test3(uint32_t *a, uint32_t *b, uint32_t ndigits);

void mul_Test1(void);
void mul_Test2(void);
void mul_test1(uint32_t *p, uint32_t a, uint32_t b);
void mul_test2(uint64_t *p, uint64_t *x, uint64_t *y);
void mul_test3(uint64_t *p, uint64_t *x, uint64_t *y);

//uint32_t sub_test1(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits);
//uint32_t sub_test2(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits);
uint32_t sub_test3(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits);
uint32_t sub_test4(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits);
uint32_t sub_test5(uint32_t w[], const uint32_t u[], const uint32_t v[], size_t ndigits);
void sub_Test1(void);
void sub_Test2(void);

void montmult_test1(uint32_t *U, uint32_t *A, uint32_t *B);
void montmult_test2(uint32_t *U, uint32_t *A,  uint32_t *B);

void montmult_Test1(void);
void montmult_Test2(void);

int main()
{
  srand(time(0));

  add_Test1();
  //add_Test2();

  comp_Test1();

  mul_Test1();
  //mul_Test2();

  sub_Test1();
  //sub_Test2();

  montmult_Test1();
  //montmult_Test2();

}

void comp_Test1()
{
  uint32_t a[8], b[8];
  double time_taken1=0.0, time_taken2=0.0, time_taken3=0.0, time_taken4=0.0;
  uint32_t i, r;
  clock_t start, end;
  uint32_t ntests = NTESTS;

  for (i=0; i<ntests; i++){
    init_u256(a);
    init_u256(b);

    start = clock();
    r = comp_test1(a,b,8);
    time_taken1 += clock() - start;
   
    start = clock();
    r = comp_test2(a,b,8);
    time_taken2 += clock() - start;
   
    start = clock();
    r += comp_test3(a,b,8);
    time_taken3 += clock() - start;
  }
   
  printf("comp T1 : %f, comp T2: %f, comp T3: %f\n",
    ((double)time_taken1)/CLOCKS_PER_SEC,
    ((double)time_taken2)/CLOCKS_PER_SEC,
    ((double)time_taken3)/CLOCKS_PER_SEC);
}

int32_t comp_test1(uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t gt=0, lt=0, ndigits2 = ndigits/2;
  uint64_t *dA = (uint64_t *)a;
  uint64_t *dB = (uint64_t *)b;

  while(ndigits2--){
    gt = (dA[ndigits2] > dB[ndigits2]);
    lt = (dA[ndigits2] < dB[ndigits2]);
    if (gt) return 1;
    if (lt) return -1;
  }
  return 0;
}

int32_t comp_test3(uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t gt=0, lt=0;

  while(ndigits--){
    gt |= (a[ndigits] > b[ndigits]);
    lt |= (a[ndigits] < b[ndigits]);
  }
  if (gt) return 1;
  if (lt) return -1;
  return 0;
}
int32_t comp_test2(uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t gt=0, lt=0;

  while(ndigits--){
    gt = (a[ndigits] > b[ndigits]);
    lt = (a[ndigits] < b[ndigits]);
    if (gt) return 1;
    if (lt) return -1;
  }
  return 0;
}
void add_Test1()
{
  uint32_t a[8], b[8], c[8];
  uint32_t carry;
  uint32_t i;
  clock_t start, end;
  double time_taken1=0.0, time_taken2=0.0, time_taken3=0.0, time_taken4=0.0, time_taken5=0.0;
  uint32_t ntests = NTESTS;

  for (i=0; i< ntests; i++){
    init_u256(a);
    init_u256(b);
   
    start = clock();
    carry = add_test1(c, a,b, 8);
    time_taken1 += clock() - start;

    start = clock();
    carry = add_test2(c, a,b,8);
    time_taken2 += clock() - start;

    start = clock();
    carry = add_test3(c, a,b,8);
    time_taken3 += clock() - start;

    start = clock();
    carry = add_test4(c, a,b,8);
    time_taken4 += clock() - start;

    start = clock();
    carry = add_test5(c, a,b,8);
    time_taken5 += clock() - start;
  }

  printf("add T1 : %f, add T2: %f, add T3: %f, add T4: %f, add T5: %f\n",
    ((double)time_taken1)/CLOCKS_PER_SEC,
    ((double)time_taken2)/CLOCKS_PER_SEC,
    ((double)time_taken3)/CLOCKS_PER_SEC,
    ((double)time_taken4)/CLOCKS_PER_SEC,
    ((double)time_taken5)/CLOCKS_PER_SEC);

}

void add_Test2()
{
  uint32_t a[8], b[8], c1[8], c2[8], c3[8], c4[8], c5[8];
  uint32_t carry1, carry2, carry3, carry4, carry5;
  uint32_t i,j, nerrors1=0, nerrors2=0, nerrors3=0, nerrors4=0,error=0;
  uint32_t ntests = NTESTS;

   
  for (i=0; i< ntests; i++){
    init_u256(a);
    init_u256(b);

    carry1 = add_test1(c1,a,b,8);
    carry2 = add_test2(c2,a,b,8);
    carry3 = add_test3(c3,a,b,8);
    carry4 = add_test4(c4,a,b,8);
    carry5 = add_test5(c5,a,b,8);
    error=0;

    if ( comp_u256(c1,c2) || (carry1 != carry2)){
      nerrors1++;
      error=1;
    }
    if ( comp_u256(c1,c3) || (carry1 != carry3)){
      nerrors2++;
      error=1;
    }
    if ( comp_u256(c1,c4) || (carry1 != carry4)){
      nerrors3++;
      error=1;
    }
    if ( comp_u256(c1,c5) || (carry1 != carry5)){
      nerrors4++;
      error=1;
    }

  }

  printf("Add N errors1 : %u, N errors2 : %u, N errors3 : %u, Nerrors4: %u\n", nerrors1, nerrors2, nerrors3, nerrors4);

}

void sub_Test1()
{
  uint32_t a[8], b[8], c[8];
  uint32_t i;
  clock_t start, end;
  double time_taken1=0.0, time_taken2=0.0, time_taken3=0.0, time_taken4=0.0, time_taken5=0.0;
  uint32_t ntests = NTESTS;

  for (i=0; i< ntests; i++){
    init_u256(a);
    init_u256(b);
   
    /*
    start = clock();
    for (i=0; i< 1<<31; i++){
      carry = sub_test1(c, a,b, 8);
    }
    time_taken1 = clock() - start;
  
    start = clock();
    for (i=0; i< 1<<31; i++){
      sub_test2(c, a,b,8);
    }
    time_taken2 = clock() - start;
    */
  
    start = clock();
    sub_test3(c, a,b,8);
    time_taken3 += clock() - start;
  
    start = clock();
    sub_test4(c, a,b,8);
    time_taken4 += clock() - start;
  
    start = clock();
    sub_test5(c, a,b,8);
    time_taken5 += clock() - start;
  }

  printf("sub T3: %f, sub T4: %f, sub T5: %f\n",
    ((double)time_taken3)/CLOCKS_PER_SEC,
    ((double)time_taken4)/CLOCKS_PER_SEC,
    ((double)time_taken5)/CLOCKS_PER_SEC);

}

void sub_Test2()
{
  uint32_t a[8], b[8], c1[8], c2[8], c3[8], c4[8], c5[8];
  uint32_t i,j, nerrors1=0, nerrors2=0, nerrors3=0, nerrors4=0,error=0;
  uint32_t ntests =NTESTS; 
   
  for (i=0; i< ntests; i++){
    init_u256(a);
    init_u256(b);

    //borrow1 = sub_test1(c1,a,b,8);
    sub_test5(c5,a,b,8);
    //sub_test2(c2,a,b,8);
    sub_test3(c3,a,b,8);
    sub_test4(c4,a,b,8);
    error=0;

    /*
    if ( comp_u256(c5,c2) ) {
      nerrors1++;
      error=1;
    }
    */
    if ( comp_u256(c5,c3) ) {
      nerrors2++;
      error=1;
    }
    if ( comp_u256(c5,c4) ){
      nerrors3++;
      error=1;
    }

  }

  printf("SUB N  N errors2 : %u, N errors3 : %u, Nerrors4: %u\n", nerrors2, nerrors3, nerrors4);

}


void mul_test1(uint32_t *p, uint32_t x, uint32_t y)
{
 /* Use a 64-bit temp for product */
 uint64_t t = (uint64_t)x * (uint64_t)y;
 /* then split into two parts */
 p[1] = (uint32_t)(t >> 32);
 p[0] = (uint32_t)(t & 0xFFFFFFFF);

}

void mul_test2(uint64_t *p, uint64_t *x, uint64_t *y)
{
 p[0] = _mulx_u64(x[0],y[0],&p[1]);
}

void mul_test3(uint64_t *p, uint64_t *a, uint64_t *b)
{
 __asm__(
	 "   mulq  %[b]\n"
		 :"=d"(p[1]), "=a"(p[0])
		 :"1"(a[0]), [b]"rm"(b[0]));
}

void mul_Test1()
{
  uint32_t a[8], b[8], c[8];
  uint64_t *dA, *dB, *dC;
  uint32_t i;
  clock_t start, end;
  double time_taken2=0.0, time_taken3=0.0, time_taken4=0.0, time_taken5=0.0;
  uint32_t ntests = NTESTS;

  dA = (uint64_t *)a;
  dB = (uint64_t *)b;
  dC = (uint64_t *)c;

  for (i=0; i<ntests;i++){
    init_u256(a);
    init_u256(b);
   
    start = clock();
    mul_test2(dC, dA,dB);
    time_taken2 += clock() - start;

    start = clock();
    mul_test3(dC, dA,dB);
    time_taken3 += clock() - start;
  }

  printf("mul T2: %f, mul T3: %f\n",
    ((double)time_taken2)/CLOCKS_PER_SEC,
    ((double)time_taken3)/CLOCKS_PER_SEC);

}

void mul_Test2()
{
  uint32_t a[8], b[8], c1[8], c2[8], c3[8];
  uint32_t i,j, nerrors1=0, error=0;
  uint64_t *dA, *dB, *dC2, *dC3;
  uint32_t ntests = NTESTS;

  dA = (uint64_t *)a;
  dB = (uint64_t *)b;
  dC2 = (uint64_t *)c2;
  dC3 = (uint64_t *)c3;

  for (i=0; i< ntests; i++){
    init_u256(a);
    init_u256(b);
   
    mul_test2(dC2,dA, dB);
    mul_test3(dC3,dA,dB);
    error=0;

    if ( comp_u256(c2,c3))
      nerrors1++;
      error=1;
    }

  printf("MUL N errors1 : %u\n", nerrors1);

}


int comp_u256(uint32_t *x, uint32_t *y)
{
  uint32_t i;

  uint64_t *dX = (uint64_t *)x;
  uint64_t *dY = (uint64_t *)y;

  if (dX[3]> dY[3]) return 1;
  if (dX[3]< dY[3]) return -1;

  if (dX[2]> dY[2]) return 1;
  if (dX[2]< dY[2]) return -1;

  if (dX[1]> dY[1]) return 1;
  if (dX[1]< dY[1]) return -1;

  if (dX[0]> dY[0]) return 1;
  if (dX[0]< dY[0]) return -1;

  return 0;
}

void init_u256(uint32_t *x)
{
  uint32_t i;
  for (i=0; i<8; i++){
   x[i] = rand();
  }
  x[7] &= 0xFFFFFF;
}

void print_u256(char *s,uint32_t *x)
{
  uint32_t i;
  printf("%s",s);
  printf("0x");
  for (i=0; i< 8; i++){
   printf("%x ", x[i]);
  }
  printf("\n");
}

uint32_t add_test1(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t carry=0;
  uint32_t i;
  
  uint64_t *dA = (uint64_t *)a;
  uint64_t *dB = (uint64_t *)b;
  uint64_t *dC = (uint64_t *)c;

  for (i=0; i<ndigits/2;i++){
    _addcarry_u64(carry, dA[i] , dB[i], &dC[i]);
  }

  return carry;
}

uint32_t add_test2(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t carry=0;
  
  const __uint128_t *dA = (__uint128_t *)a;
  const __uint128_t *dB = (__uint128_t *)b;
  __uint128_t *dC = (__uint128_t *)c;

  dC[0] = dA[0] + dB[0];
  carry = (dC[0] < dA[0]);

  dC[1] = dA[1] + carry;
  carry = (dC[1] < carry);
  dC[1] += dB[1];
  carry += (dC [1]< dB[1]);

  return carry;
}

uint32_t add_test3(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t carry=0;

  const uint64_t *dA = (uint64_t *)a;
  const uint64_t *dB = (uint64_t *)b;
  uint64_t *dC = (uint64_t *)c;
  uint64_t tmp = dA[0];

  dC[0] = dA[0] + dB[0];
  carry = (dC[0] < tmp);

  tmp = dB[1];
  dC[1] = dA[1] + carry;
  carry = (dC[1] < carry);
  dC[1] += tmp;
  carry += (dC[1] < tmp);

  tmp = dB[2];
  dC[2] = dA[2] + carry;
  carry = (dC[2] < carry);
  dC[2] += tmp;
  carry += (dC[2] < tmp);

  tmp = dB[3];
  dC[3] = dA[3] + carry;
  carry = (dC[3] < carry);
  dC[3] += tmp;
  carry += (dC[3] < tmp);

 return carry;
}

uint64_t add_test6(uint64_t *c, uint64_t *a, uint64_t *b)
{
  uint64_t carry=0;

  const uint64_t *dA = (uint64_t *)a;
  const uint64_t *dB = (uint64_t *)b;
  uint64_t *dC = (uint64_t *)c;
  uint64_t tmp = dA[0];

  dC[0] = dA[0] + dB[0];
  carry = (dC[0] < tmp);

 return carry;
}


uint32_t add_test4(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t carry=0;
  
  c[0] = a[0] + b[0];
  carry = (c[0] < a[0]);

  c[1] = a[1] + carry;
  carry = (c[1] < carry);
  c[1] += b[1];
  carry += (c[1] < b[1]);

  c[2] = a[2] + carry;
  carry = (c[2] < carry);
  c[2] += b[2];
  carry += (c[2] < b[2]);

  c[3] = a[3] + carry;
  carry = (c[3] < carry);
  c[3] += b[3];
  carry += (c[3] < b[3]);

  c[4] = a[4] + carry;
  carry = (c[4] < carry);
  c[4] += b[4];
  carry += (c[4] < b[4]);

  c[5] = a[5] + carry;
  carry = (c[5] < carry);
  c[5] += b[5];
  carry += (c[5] < b[5]);

  c[6] = a[6] + carry;
  carry = (c[6] < carry);
  c[6] += b[6];
  carry += (c[6] < b[6]);

  c[7] = a[7] + carry;
  carry = (c[7] < carry);
  c[7] += b[7];
  carry += (c[7] < b[7]);

  return carry;
}

uint32_t add_test5(uint32_t w[], uint32_t u[], uint32_t v[], size_t ndigits)
{

 uint32_t k;
 size_t j;

 /* Step A1. Initialise */
 k = 0;

 for (j = 0; j < ndigits; j++) {
  /* Step A2. Add digits w_j = (u_j + v_j + k)
   Set k = 1 if carry (overflow) occurs
  */
  w[j] = u[j] + k;
  if (w[j] < k) k = 1; 
  else k = 0;

  w[j] += v[j];
  if (w[j] < v[j]) k++;

 } /* Step A3. Loop on j */

 return k; /* w_n = k */
}

#if 0
uint32_t sub_test1(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t carry=0;
  uint32_t i;
  
  uint64_t *dA = (uint64_t *)a;
  uint64_t *dB = (uint64_t *)b;
  uint64_t *dC = (uint64_t *)c;

  for (i=0; i<ndigits/2;i++){
    _subcarry_u64(carry, dA[i] , dB[i], &dC[i]);
  }

  return carry;
}

uint32_t sub_test2(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t borrow=0;
  
  const __uint128_t *dA = (__uint128_t *)a;
  const __uint128_t *dB = (__uint128_t *)b;
  __uint128_t *dC = (__uint128_t *)c;

  dC[0] = dA[0] - dB[0];
  borrow = (dC[0] > MAX_128DIGIT - dB[0] );

  dC[1] = dA[1] - borrow;
  borrow = (dC[1] > MAX_128DIGIT - borrow);
  dC[1] -= dB[1];
  borrow += (dC [1]> MAX_128DIGIT - dB[1]);

  return borrow;
}

#endif
uint32_t sub_test3(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t carry=0;

  const uint64_t *dA = (uint64_t *)a;
  const uint64_t *dB = (uint64_t *)b;
  uint64_t *dC = (uint64_t *)c;
  uint64_t tmp = dA[0];

  /*
  dC[0] = dA[0] - dB[0];
  borrow = (dC[0] > MAX_64DIGIT - dB[0]);

  dC[1] = dA[1] - borrow;
  borrow = (dC[1] > MAX_64DIGIT - borrow);
  dC[1] -= dB[1];
  borrow += (dC[1] > MAX_64DIGIT - dB[1]);

  dC[2] = dA[2] - borrow;
  borrow = (dC[2] > MAX_64DIGIT - borrow);
  dC[2] -= dB[2];
  borrow += (dC[2] > MAX_64DIGIT - dB[2]);

  dC[3] = dA[3] - borrow;
  borrow = (dC[3] > MAX_64DIGIT - borrow);
  dC[3] -= dB[3];
  borrow += (dC[3] > MAX_64DIGIT - dB[3]);
  */

  dC[0] = dA[0] + (NEG(dB[0])+1);
  carry = (dC[0] < tmp);

  tmp = NEG(dB[1]);
  dC[1] = dA[1] + carry;
  carry = (dC[1] < carry);
  dC[1] += tmp;
  carry += (dC[1] < tmp);

  tmp = NEG(dB[2]);
  dC[2] = dA[2] + carry;
  carry = (dC[2] < carry);
  dC[2] += tmp;
  carry += (dC[2] < tmp);

  tmp = NEG(dB[3]);
  dC[3] = dA[3] + carry;
  carry = (dC[3] < carry);
  dC[3] += tmp;
  carry += (dC[3] < tmp);

 return carry;
}

uint32_t sub_test4(uint32_t *c, uint32_t *a, uint32_t *b, uint32_t ndigits)
{
  uint32_t borrow=0;
  
  c[0] = a[0] - b[0];
  borrow = (c[0] > MAX_32DIGIT - b[0]);

  c[1] = a[1] - borrow;
  borrow = (c[1] > MAX_32DIGIT - borrow);
  c[1] -= b[1];
  borrow += (c[1] > MAX_32DIGIT - b[1]);

  c[2] = a[2] - borrow;
  borrow = (c[2] > MAX_32DIGIT - borrow);
  c[2] -= b[2];
  borrow += (c[2] > MAX_32DIGIT - b[2]);

  c[3] = a[3] - borrow;
  borrow = (c[3] > MAX_32DIGIT - borrow);
  c[3] -= b[3];
  borrow += (c[3] > MAX_32DIGIT - b[3]);

  c[4] = a[4] - borrow;
  borrow = (c[4] > MAX_32DIGIT - borrow);
  c[4] -= b[4];
  borrow += (c[4] > MAX_32DIGIT - b[4]);

  c[5] = a[5] - borrow;
  borrow = (c[5] > MAX_32DIGIT - borrow);
  c[5] -= b[5];
  borrow += (c[5] > MAX_32DIGIT - b[5]);

  c[6] = a[6] - borrow;
  borrow = (c[6] > MAX_32DIGIT - borrow);
  c[6] -= b[6];
  borrow += (c[6] > MAX_32DIGIT - b[6]);

  c[7] = a[7] - borrow;
  borrow = (c[7] > MAX_32DIGIT - borrow);
  c[7] -= b[7];
  borrow += (c[7] > MAX_32DIGIT - b[7]);

  return borrow;
}

uint32_t sub_test5(uint32_t w[], const uint32_t u[], const uint32_t v[], size_t ndigits)
{
 uint32_t k;
 size_t j;

 /* Step S1. Initialise */
 k = 0;

 for (j = 0; j < ndigits; j++)
 {
  /* Step S2. Subtract digits w_j = (u_j - v_j - k)
   Set k = 1 if borrow occurs.
  */
  w[j] = u[j] - k;
  if (w[j] > MAX_32DIGIT - k) k = 1;
  else k = 0;

  w[j] -= v[j];
  if (w[j] > MAX_32DIGIT - v[j]) k++;

 } /* Step S3. Loop on j */

 return k;

}

void montmult_test1(uint32_t *U, uint32_t *A, uint32_t *B)
{
  int i, j;
  uint32_t S, C, C1, C2, C3=0, M[2], X[2], carry;
  uint32_t T[NWORDS_256BIT_FIOS];

  memset(T, 0, sizeof(uint32_t)*(NWORDS_256BIT_FIOS));

  //print_u256("A-32: \n",A);
  //print_u256("B-32: \n",B);

  for(i=0; i<NWORDS_256BIT; i++) {
    // (C,S) = t[0] + a[0]*b[i], worst case 2 words
    mul_test1(X, A[0], B[i]); // X[Upper,Lower] = a[0]*b[i]
    C = add_test5(&S, T+0, X+0, 1); // [C,S] = t[0] + X[Lower]
    add_test5(&C, &C, X+1, 1);  // [~,C] = C + X[Upper], No carry
    //printf("1[%d]: C: %llx S: %llx\n",i,(uint64_t)C, (uint64_t)S); 

    // ADD(t[1],C)
    carry = add_test5(&T[1], &T[1], &C, 1); 
    // m = S*n'[0] mod W, where W=2^32
    // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
    mul_test1(M, S, NPrime[0]);

    // (C,S) = S + m*n[0], worst case 2 words
    mul_test1(X, M[0], N[0]); // X[Upper,Lower] = m*n[0]
    C = add_test5(&S, &S, X+0, 1); // [C,S] = S + X[Lower]
    add_test5(&C, &C, X+1, 1);  // [~,C] = C + X[Upper]
    //printf("2[%d]: C: %llx S: %llx, carry: %llx\n",i,(uint64_t)C, (uint64_t)S, (uint64_t)carry); 

    for(j=1; j<NWORDS_256BIT; j++) {
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mul_test1(X, A[j], B[i]);   // X[Upper,Lower] = a[j]*b[i], double precision
      C1 = add_test5(&S, T+j, &C, 1);  // (C1,S) = t[j] + C
      C2 = add_test5(&S, &S, X+0, 1);  // (C2,S) = S + X[Lower]
      add_test5(&C, &C1, X+1, 1);   // (~,C)  = C1 + X[Upper], doesn't produce carry
      C3 = add_test5(&C, &C, &C2, 1);    // (~,C)  = C + C2, it DOES produce carry
      //printf("3[%d-%d]: C1: %llx C: %llx S: %llx\n",i,j,(uint64_t)C3,(uint64_t) C, (uint64_t)S); 
      // ADD(t[j+1],C)
      C3 += add_test5(&C, &C, &carry, 1);    // (~,C)  = C + C2, It DOES produce carry

      carry = add_test5(&T[j+1], &T[j+1], &C, 1) + C3; 
      //printf("4[%d-%d]: C1: %llx C: %llx S: %llx, carry: %llx\n",i,j,(uint64_t) C3,(uint64_t)C, (uint64_t)T[j+1],(uint64_t)carry); 
   
      // (C,S) = S + m*n[j]
      mul_test1(X, M[0], N[j]); // X[Upper,Lower] = m*n[j]
      C = add_test5(&S, &S, X+0, 1); // [C,S] = S + X[Lower]
      add_test5(&C, &C, X+1, 1);  // [~,C] = C + X[Upper]
   
      // t[j-1] = S
      T[j-1] = S;

      //print_u256("T1 : \n",T);
    }

    // (C,S) = t[s] + C
    C = add_test5(&S, T+NWORDS_256BIT, &C, 1);
    // t[s-1] = S
    T[NWORDS_256BIT-1] = S;
    // t[s] = t[s+1] + C
    add_test5(T+NWORDS_256BIT, T+NWORDS_256BIT+1, &C, 1);
    // t[s+1] = 0
    T[NWORDS_256BIT+1] = 0;
     //print_u256("T2 : \n",T);
  }

  //print_u256("T : \n",T);

  /* Step 3: if(u>=n) return u-n else return u */
  if(comp_test3(T, (uint32_t *)N, 8) >= 0) {
    //printf("Compute Mod\n");
    sub_test5(T, T, N, NWORDS_256BIT);
  }

  memcpy(U, T, sizeof(uint32_t)*NWORDS_256BIT);
  //print_u256("U : \n",U);
}

void montmult_test2(uint32_t *U,  uint32_t *A, uint32_t *B)
{

  int i, j;
  uint32_t T[NWORDS_256BIT_FIOS+1];
  uint64_t *dA = (uint64_t *)A;
  uint64_t *dB = (uint64_t *)B;
  uint64_t *dU = (uint64_t *)U;
  uint64_t S, C, C1,C2,C3=0,carry, M[2], X[2];
  uint64_t *dNP = (uint64_t *)NPrime;
  uint64_t *dN = (uint64_t *)N;

  memset(T, 0, sizeof(uint32_t)*(NWORDS_256BIT_FIOS+1));
  uint64_t *dT = (uint64_t *)T;

  //print_u256("A-64: \n",A);
  //print_u256("B-64: \n",B);

  for(i=0; i<NWORDS_256BIT/2; i++) {
    // (C,S) = t[0] + a[0]*b[i], worst case 2 words
    mul_test2(X, &dA[0], &dB[i]); // X[Upper,Lower] = a[0]*b[i]
    C = add_test6(&S, dT+0, X+0); // [C,S] = t[0] + X[Lower]
    add_test6(&C, &C, X+1);  // [~,C] = C + X[Upper], No carry
    //printf("1[%d]: C: %llx S: %llx\n",i,(uint64_t)C, (uint64_t)S); 

    // ADD(t[1],C)
    carry = add_test6(&dT[1], &dT[1], &C); 
    //printf("a[%d]: C: %llx T[1]: %llx\n",i,(uint64_t)carry, (uint64_t)dT[1]); 
    // m = S*n'[0] mod W, where W=2^32
    // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
    mul_test2(M, &S, dNP);
    //printf("b[%d]: M: %llx, N: %llx\n",i,(uint64_t)(M[0]),(uint64_t)dN[0]);

    // (C,S) = S + m*n[0], worst case 2 words
    mul_test2(X, &M[0], dN); // X[Upper,Lower] = m*n[0]
    C = add_test6(&S, &S, X+0); // [C,S] = S + X[Lower]
    add_test6(&C, &C, X+1);  // [~,C] = C + X[Upper]
    //printf("2[%d]: C: %llx S: %llx, carry: %llx\n",i,(uint64_t)C, (uint64_t)S, (uint64_t)carry); 

    for(j=1; j<NWORDS_256BIT/2; j++) {
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mul_test2(X, &dA[j], &dB[i]);   // X[Upper,Lower] = a[j]*b[i], double precision
      C1 = add_test6(&S, dT+j, &C);  // (C1,S) = t[j] + C
      C2 = add_test6(&S, &S, X+0);  // (C2,S) = S + X[Lower]
      add_test6(&C, &C1, X+1);   // (~,C)  = C1 + X[Upper], doesn't produce carry
      C3 = add_test6(&C, &C, &C2);    // (~,C)  = C + C2, it DOES produce carry
      //printf("3[%d-%d]: C1: %llx C: %llx S: %llx\n",i,j,(uint64_t)C3,(uint64_t) C, (uint64_t)S); 
      // ADD(t[j+1],C)
      C3 += add_test6(&C, &C, &carry);    // (~,C)  = C + C2, It DOES produce carry

      //printf("c[%d-%d]: C1: %llu C: %llx T[j+1]: %llx\n",i,j,(uint64_t) C3,(uint64_t)C, (uint64_t)dT[j+1]); 
      carry = add_test6(&dT[j+1], &dT[j+1], &C) + C3; 
      //printf("4[%d-%d]: C1: %llx C: %llx S: %llx, carry: %llx\n",i,j,(uint64_t) C3,(uint64_t)C, (uint64_t)dT[j+1],(uint64_t)carry); 
   
      // (C,S) = S + m*n[j]
      mul_test2(X, M, &dN[j]); // X[Upper,Lower] = m*n[j]
      C = add_test6(&S, &S, X+0); // [C,S] = S + X[Lower]
      add_test6(&C, &C, X+1);  // [~,C] = C + X[Upper]
   
      // t[j-1] = S
      dT[j-1] = S;
      
      //print_u256("T1 : \n",T);
    }

    // (C,S) = t[s] + C
    C = add_test6(&S, dT+NWORDS_256BIT/2, &C);
    // t[s-1] = S
    dT[NWORDS_256BIT/2-1] = S;
    // t[s] = t[s+1] + C
    add_test6(dT+NWORDS_256BIT/2, dT+NWORDS_256BIT/2+1, &C);
    // t[s+1] = 0
    dT[NWORDS_256BIT/2+1] = 0;
    //print_u256("T2 : \n",T);
  }

  //print_u256("T : \n",T);
  /* Step 3: if(u>=n) return u-n else return u */
  if(comp_test3(T, N, 8) >= 0) {
    //printf("Compute Mod\n");
    sub_test3(T, T, N, NWORDS_256BIT);
  }

  memcpy(U, T, sizeof(uint32_t)*NWORDS_256BIT);
  //print_u256("U : \n",U);
}


void montmult_Test1()
{
  uint32_t a[8], b[8], c[8];
  uint32_t i;
  clock_t start, end;
  double time_taken1, time_taken2;
  uint32_t ntests = NTESTS;

  for (i=0; i<ntests;i++){
    init_u256(a);
    init_u256(b);
   
    start = clock();
    montmult_test1(c, a,b);
    time_taken1 += clock() - start;

    start = clock();
    montmult_test2(c, a,b);
    time_taken2 += clock() - start;
  }

  printf("montmult T1: %f, montmult T2: %f\n",
    ((double)time_taken1)/CLOCKS_PER_SEC,
    ((double)time_taken2)/CLOCKS_PER_SEC);

}

void montmult_Test2()
{
  uint32_t a[8], b[8], c1[8], c2[8];
  uint32_t i,j, nerrors1=0, nerrors2=0, error=0;
  uint32_t ntests = 1<<25;

   
  for (i=0; i< ntests; i++){
    init_u256(a);
    init_u256(b);

    montmult_test1(c1,a,b);
    montmult_test2(c2,a,b);
    error=0;

    if ( comp_u256(c1,c2) ) {
      nerrors2++;
      error=1;
    }
  }

  printf("MONTMULT  N errors : %u/%u\n",nerrors2,ntests);

}

