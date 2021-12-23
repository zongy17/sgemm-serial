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
#define SET_P_BLOCK_SIZE 8
#endif 
#if !defined(SET_J_BLOCK_SIZE)
#define SET_J_BLOCK_SIZE 64
#endif
#if !defined(SET_I_BLOCK_SIZE)
#define SET_I_BLOCK_SIZE 32
#endif

float C_local[SET_P_BLOCK_SIZE*SET_J_BLOCK_SIZE] __attribute__((aligned(64)));
float A_local[SET_P_BLOCK_SIZE*SET_I_BLOCK_SIZE] __attribute__((aligned(64)));
float B_local[SET_I_BLOCK_SIZE*SET_J_BLOCK_SIZE] __attribute__((aligned(64)));

static void copy_into_MxN_nopadding(int n, int realM, int realN, int setM, const float* __restrict__ array, float* __restrict__ array_local) {
  for (int local_col = 0; local_col < realN; local_col++){
    for (int local_row = 0; local_row < realM; local_row++)
      array_local[local_row] = array[local_row];
    array_local += setM;
    array += n;
  }
}

// 原来是IxJ的矩阵，转置后当作JxI的矩阵
static void copy_transpose_IxJ_into_JxI_nopadding(int n, int realI, int realJ, int setI, int setJ, const float* __restrict__ B, float* __restrict__ B_local) {
  for (int new_local_col = 0; new_local_col < realI; new_local_col++){
    for (int new_local_row = 0; new_local_row < realJ; new_local_row++)
      B_local[new_local_row] = B[new_local_row*n];
    B_local += setJ;
    B += 1;
  }
}


void square_sgemm (int n, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {

  for (int j_block = 0; j_block < n; j_block += SET_J_BLOCK_SIZE){//对于B而言的水平划分
    int REAL_J_BLOCK_SIZE = min(SET_J_BLOCK_SIZE, n - j_block);

    for (int i_block = 0; i_block < n; i_block += SET_I_BLOCK_SIZE){//对于B而言的垂直划分
      int REAL_I_BLOCK_SIZE = min(SET_I_BLOCK_SIZE, n - i_block);

      // copy_into_MxN_nopadding(n, REAL_I_BLOCK_SIZE, REAL_J_BLOCK_SIZE, SET_I_BLOCK_SIZE, B + j_block*n + i_block, B_local);
      // 拷贝并转置B的一部分放到B_local里
      copy_transpose_IxJ_into_JxI_nopadding(n, REAL_I_BLOCK_SIZE, REAL_J_BLOCK_SIZE, SET_I_BLOCK_SIZE, SET_J_BLOCK_SIZE, B + j_block*n + i_block, B_local);

      for (int p_block = 0; p_block < n; p_block += SET_P_BLOCK_SIZE) {
        int REAL_P_BLOCK_SIZE = min(SET_P_BLOCK_SIZE, n - p_block);

        copy_into_MxN_nopadding(n, REAL_P_BLOCK_SIZE, REAL_I_BLOCK_SIZE, SET_P_BLOCK_SIZE, A + i_block*n + p_block, A_local);

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
        for (int i = 0; i < REAL_I_BLOCK_SIZE; i++){
          for ( j = 0; j < ((REAL_J_BLOCK_SIZE)&(~7)); j+=8){//这里还是按照没转置的观点去遍历
            // register float b0 = B_local[j+i*SET_J_BLOCK_SIZE];//转置前是B_local[i+j*SET_I_BLOCK_SIZE]
            register float b0 = B_local_ptr[j  ];
            register float b1 = B_local_ptr[j+1];
            register float b2 = B_local_ptr[j+2];
            register float b3 = B_local_ptr[j+3];
            register float b4 = B_local_ptr[j+4];
            register float b5 = B_local_ptr[j+5];
            register float b6 = B_local_ptr[j+6];
            register float b7 = B_local_ptr[j+7]; 
            for (int p = 0; p < REAL_P_BLOCK_SIZE; p++){
              // C_local[j*SET_P_BLOCK_SIZE + p] += A_local[i*SET_P_BLOCK_SIZE + p] * b0;
              C_local_ptr[                     p] += A_local_ptr[p] * b0;
              C_local_ptr[  SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b1;
              C_local_ptr[2*SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b2;
              C_local_ptr[3*SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b3;
              C_local_ptr[4*SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b4;
              C_local_ptr[5*SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b5;
              C_local_ptr[6*SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b6;
              C_local_ptr[7*SET_P_BLOCK_SIZE + p] += A_local_ptr[p] * b7;
            }
            C_local_ptr += 8*SET_P_BLOCK_SIZE;
          }
          for ( ; j < REAL_J_BLOCK_SIZE; j++){
            register float b0 = B_local_ptr[j];
            for (int p = 0; p < REAL_P_BLOCK_SIZE; p++)
              C_local_ptr[p] += A_local_ptr[p] * b0;
            C_local_ptr += SET_P_BLOCK_SIZE;
          }
          C_local_ptr = C_local;//C重新归位
          B_local_ptr += SET_J_BLOCK_SIZE;
          A_local_ptr += SET_P_BLOCK_SIZE;
        }

        // 计算完拷贝回去
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

void square_sgemm_naive (int n, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C)
// void square_sgemm (int n, float* A, float* B, float* C)
{
  // /* For each row i of A */
  // for (int i = 0; i < n; ++i)
  //   /* For each column j of B */
  //   for (int j = 0; j < n; ++j) 
  //   {
  //     /* Compute C(i,j) */
  //     float cij = C[i+j*n];
  //     for( int k = 0; k < n; k++ )
	//        cij += A[i+k*n] * B[k+j*n];
  //     C[i+j*n] = cij;
  //   }

  // 循环变换(no intrinsics)
  int j, i, p;
  // for ( j = 0; j < n; j++)//for each colum j of B
  //   for ( i = 0; i < n; i++){//for each row i of B
  //     register float b0 = B(i,j);
  //     for ( p = 0; p < n; p++)
  //       C(p,j  ) += A(p,i ) * b0;
  //   }

  for ( j = 0; j < ((n)&(~3)); j+=4)//for each colum j of B
    for ( i = 0; i < n; i++){//for each row i of B
      register float b0 = B(i,j);
      register float b1 = B(i,j+1);
      register float b2 = B(i,j+2);
      register float b3 = B(i,j+3);
      for ( p = 0; p < n; p++){
        C(p,j  ) += A(p,i ) * b0;
        C(p,j+1) += A(p,i  ) * b1;
        C(p,j+2) += A(p,i  ) * b2;
        C(p,j+3) += A(p,i  ) * b3;
      }
    }
  for ( ; j < n; j++)//余下的列
    for ( i = 0; i < n; i++){//for each row i of B
      register float b0 = B(i,j);
      for ( p = 0; p < n; p++){
        C(p,j  ) += A(p,i  ) * b0;
      }
    }
}

