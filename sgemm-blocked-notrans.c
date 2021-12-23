const char* sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#include <arm_neon.h>
// #include "arm_neon.h"

#define min(a,b) (((a)<(b))?(a):(b))

#define A(i,j) A[ (j)*n + (i) ]
#define B(i,j) B[ (j)*n + (i) ]
#define C(i,j) C[ (j)*n + (i) ]

#if !defined(SET_P_BLOCK_SIZE)
#define SET_P_BLOCK_SIZE 124
#endif 
#if !defined(SET_J_BLOCK_SIZE)
#define SET_J_BLOCK_SIZE 96
#endif
#if !defined(SET_I_BLOCK_SIZE)
#define SET_I_BLOCK_SIZE 100
#endif

float C_local[SET_P_BLOCK_SIZE*SET_J_BLOCK_SIZE] __attribute__((aligned(64)));
float A_local[SET_P_BLOCK_SIZE*SET_I_BLOCK_SIZE] __attribute__((aligned(64)));
float B_local[SET_I_BLOCK_SIZE*SET_J_BLOCK_SIZE] __attribute__((aligned(64)));

static void copy_into_MxN_nopadding(int n, int realM, int realN, int setM, \
      const float* __restrict__ array, float* __restrict__ array_local) {
  for (int local_col = 0; local_col < realN; local_col++){
    for (int local_row = 0; local_row < realM; local_row++)
      array_local[local_row] = array[local_row];
    array_local += setM;
    array += n;
  }
}

void square_sgemm (int n, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {

  for (int j_block = 0; j_block < n; j_block += SET_J_BLOCK_SIZE){//对于B而言的水平划分
    int REAL_J_BLOCK_SIZE = min(SET_J_BLOCK_SIZE, n - j_block);

    for (int i_block = 0; i_block < n; i_block += SET_I_BLOCK_SIZE){//对于B而言的垂直划分
      int REAL_I_BLOCK_SIZE = min(SET_I_BLOCK_SIZE, n - i_block);

      copy_into_MxN_nopadding(n, REAL_I_BLOCK_SIZE, REAL_J_BLOCK_SIZE, SET_I_BLOCK_SIZE,\
                              B + j_block*n + i_block, B_local);

      for (int p_block = 0; p_block < n; p_block += SET_P_BLOCK_SIZE) {
        int REAL_P_BLOCK_SIZE = min(SET_P_BLOCK_SIZE, n - p_block);

        copy_into_MxN_nopadding(n, REAL_P_BLOCK_SIZE, REAL_I_BLOCK_SIZE, SET_P_BLOCK_SIZE,\
                                A + i_block*n + p_block, A_local);

        // local_C清零
        float * C_local_ptr = C_local;
        for (int j = 0; j < REAL_J_BLOCK_SIZE; j++){//拷贝的时候是部分
          for (int p = 0; p < REAL_P_BLOCK_SIZE; p++)
            C_local_ptr[p] = 0.0;
          C_local_ptr += SET_P_BLOCK_SIZE;//而指针前进的时候是全步长！
        }

        // 计算
        float * B_local_ptr = B_local;
        float * A_local_ptr = A_local;
        C_local_ptr = C_local;
        int j;
        for ( j = 0; j < ((REAL_J_BLOCK_SIZE)&(~3)); j+=4){//计算的时候是部分
          for (int i = 0; i < REAL_I_BLOCK_SIZE; i++){
            register float b0 = B_local_ptr[i];
            register float b1 = B_local_ptr[SET_I_BLOCK_SIZE + i];
            register float b2 = B_local_ptr[2*SET_I_BLOCK_SIZE + i];
            register float b3 = B_local_ptr[3*SET_I_BLOCK_SIZE + i];
            for (int p = 0; p < REAL_P_BLOCK_SIZE; p++){
              C_local_ptr[p] += A_local_ptr[p] * b0;
              C_local_ptr[SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b1;
              C_local_ptr[2*SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b2;
              C_local_ptr[3*SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b3;
            }
            A_local_ptr += SET_P_BLOCK_SIZE;//而指针前进的时候是全步长！
          }
          A_local_ptr = A_local;//A重新归位
          B_local_ptr += 4*SET_I_BLOCK_SIZE;
          C_local_ptr += 4*SET_P_BLOCK_SIZE;
        }
        for ( ; j < REAL_J_BLOCK_SIZE; j++){//计算的时候是部分
          for (int i = 0; i < REAL_I_BLOCK_SIZE; i++){
            register float b0 = B_local_ptr[i];
            for (int p = 0; p < REAL_P_BLOCK_SIZE; p++){
              C_local_ptr[p] += A_local_ptr[p] * b0;
            }
            A_local_ptr += SET_P_BLOCK_SIZE;//而指针前进的时候是全步长！
          }
          A_local_ptr = A_local;//A重新归位
          B_local_ptr += SET_I_BLOCK_SIZE;
          C_local_ptr += SET_P_BLOCK_SIZE;
        }

        // 计算完拷贝回去
        // for (int j = 0; j < REAL_J_BLOCK_SIZE; j++){
        //   for (int i = 0; i < REAL_I_BLOCK_SIZE; i++){
        //     register float b0 = B_local[j*SET_I_BLOCK_SIZE + i];
        //     for (int p = 0; p < REAL_P_BLOCK_SIZE; p++)
        //       C[(j_block+j)*n + p_block + p] += A_local[i*SET_P_BLOCK_SIZE + p] * b0;
        //   }
        // }
        C += j_block*n + p_block;
        C_local_ptr = C_local;
        for (int j = 0; j < REAL_J_BLOCK_SIZE; j++){//拷贝的时候是部分
          for (int p = 0; p < REAL_P_BLOCK_SIZE; p++)
            C[p] += C_local_ptr[p];
          C += n;
          C_local_ptr += SET_P_BLOCK_SIZE;//而指针前进的时候是全步长！
        }
        C -= (j_block+REAL_J_BLOCK_SIZE)*n + p_block;
      }
    }
  }
}