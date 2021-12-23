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

#define SET_P_BLOCK_SIZE 1024
#define SET_J_BLOCK_SIZE 1024
#define SET_I_BLOCK_SIZE 256

#define KERNEL_SIZE_ROW 8
#define KERNEL_SIZE_COL 8

// float C_local[SET_P_BLOCK_SIZE*SET_J_BLOCK_SIZE] __attribute__((aligned(64)));
float A_local[SET_P_BLOCK_SIZE*SET_I_BLOCK_SIZE] __attribute__((aligned(64)));
float B_local[SET_I_BLOCK_SIZE*SET_J_BLOCK_SIZE] __attribute__((aligned(64)));
float tmp[KERNEL_SIZE_ROW*KERNEL_SIZE_COL] __attribute__((aligned(64)));
float C_buffer[KERNEL_SIZE_ROW*KERNEL_SIZE_COL] __attribute__((aligned(64)));//只是用来padding时缓冲一下累加值

static void copy_PxI_nopadding(int n, int REAL_I_BLOCK_SIZE, \
            const float* __restrict__ A, float* __restrict__ A_local) {
  for (int local_col = 0; local_col < REAL_I_BLOCK_SIZE; local_col++){
    for (int local_row = 0; local_row < KERNEL_SIZE_ROW; local_row++)
      A_local[local_row] = A(local_row, 0);//相当于把原A中一个块在内存中离散的数据拷贝成A_local中连续的一大片
    A_local += KERNEL_SIZE_ROW;
    A += n;
  }
}

static void copy_A_into_PxI(int n, int REAL_P_BLOCK_SIZE, int REAL_I_BLOCK_SIZE, \
                      const float* __restrict__ A, float* __restrict__ A_local) {
  int part = REAL_P_BLOCK_SIZE / KERNEL_SIZE_ROW;
  int remain_rows = REAL_P_BLOCK_SIZE % KERNEL_SIZE_ROW;
  for (int pa = 0; pa < part; pa++){
    copy_PxI_nopadding(n, REAL_I_BLOCK_SIZE, A, A_local);
    A_local += KERNEL_SIZE_ROW * REAL_I_BLOCK_SIZE;
    A += KERNEL_SIZE_ROW;//指针指向下一个块
  }
  if (remain_rows > 0) {//余下还有
    for (int local_col = 0; local_col < REAL_I_BLOCK_SIZE; local_col++){
      for (int local_row = 0; local_row < remain_rows; local_row++)
        A_local[local_row] = A(local_row, 0);
      for (int local_row = remain_rows; local_row < KERNEL_SIZE_ROW; local_row++)
        A_local[local_row] = 0.0;
      A_local += KERNEL_SIZE_ROW;
      A += n;
    }
  }
}

static void copy_transpose_IxJ_nopadding(int n, int REAL_I_BLOCK_SIZE, \
            const float* __restrict__ B, float* __restrict__ B_local) {
  for (int local_col = 0; local_col < REAL_I_BLOCK_SIZE; local_col++){
    for (int local_row = 0; local_row < KERNEL_SIZE_COL; local_row++)
      B_local[local_row] = B(0, local_row);
    B_local += KERNEL_SIZE_COL;
    B += 1;
  }
}

static void copy_transpose_B_into_IxJ(int n, int REAL_I_BLOCK_SIZE, int REAL_J_BLOCK_SIZE,\
                               const float* __restrict__ B, float* __restrict__ B_local) {
  int part = REAL_J_BLOCK_SIZE / KERNEL_SIZE_COL;
  int remain_cols = REAL_J_BLOCK_SIZE % KERNEL_SIZE_COL;
  for (int pa = 0; pa < part; pa++){
    copy_transpose_IxJ_nopadding(n, REAL_I_BLOCK_SIZE, B, B_local);
    B_local += KERNEL_SIZE_COL * REAL_I_BLOCK_SIZE;
    B += KERNEL_SIZE_COL * n;
  }
  if (remain_cols > 0) {
    for (int local_col = 0; local_col < REAL_I_BLOCK_SIZE; local_col++) {
      for (int local_row = 0; local_row < remain_cols; local_row++)
        B_local[local_row] = B(0, local_row);
      for (int local_row = remain_cols; local_row < KERNEL_SIZE_COL; local_row++)
        B_local[local_row] = 0.0;
      B_local += KERNEL_SIZE_COL;
      B += 1;
    }
  }
}

static void sgemm_kernel(int REAL_I_BLOCK_SIZE, const float* __restrict__ a, const float* \
    __restrict__ b, float* __restrict__ CorCBuffer, int C_direct, int row_CorCBuffer) {

  float32x4_t c00 = {0}, c01 = {0}, c02 = {0}, c03 = {0};
  float32x4_t c04 = {0}, c05 = {0}, c06 = {0}, c07 = {0};
  float32x4_t c40 = {0}, c41 = {0}, c42 = {0}, c43 = {0};
  float32x4_t c44 = {0}, c45 = {0}, c46 = {0}, c47 = {0};
  for (int l = 0; l < REAL_I_BLOCK_SIZE; l++) {
    float32x4_t value_a0 = vld1q_f32(a + KERNEL_SIZE_ROW*l    );
    float32x4_t value_a4 = vld1q_f32(a + KERNEL_SIZE_ROW*l + 4);
    float32x4_t value_b0 = vld1q_f32(b + KERNEL_SIZE_COL*l    );
    float32x4_t value_b4 = vld1q_f32(b + KERNEL_SIZE_COL*l + 4);
    c00 = vmlaq_laneq_f32(c00, value_a0, value_b0, 0);
    c01 = vmlaq_laneq_f32(c01, value_a0, value_b0, 1);
    c02 = vmlaq_laneq_f32(c02, value_a0, value_b0, 2);
    c03 = vmlaq_laneq_f32(c03, value_a0, value_b0, 3);
    c04 = vmlaq_laneq_f32(c04, value_a0, value_b4, 0);
    c05 = vmlaq_laneq_f32(c05, value_a0, value_b4, 1);
    c06 = vmlaq_laneq_f32(c06, value_a0, value_b4, 2);
    c07 = vmlaq_laneq_f32(c07, value_a0, value_b4, 3);
    c40 = vmlaq_laneq_f32(c40, value_a4, value_b0, 0);
    c41 = vmlaq_laneq_f32(c41, value_a4, value_b0, 1);
    c42 = vmlaq_laneq_f32(c42, value_a4, value_b0, 2);
    c43 = vmlaq_laneq_f32(c43, value_a4, value_b0, 3);
    c44 = vmlaq_laneq_f32(c44, value_a4, value_b4, 0);
    c45 = vmlaq_laneq_f32(c45, value_a4, value_b4, 1);
    c46 = vmlaq_laneq_f32(c46, value_a4, value_b4, 2);
    c47 = vmlaq_laneq_f32(c47, value_a4, value_b4, 3);
  }
  // 存到临时变量
  vst1q_f32(tmp     , c00);
  vst1q_f32(tmp + 4 , c40);
  vst1q_f32(tmp + 8 , c01);
  vst1q_f32(tmp + 12, c41);
  vst1q_f32(tmp + 16, c02);
  vst1q_f32(tmp + 20, c42);
  vst1q_f32(tmp + 24, c03);
  vst1q_f32(tmp + 28, c43);
  vst1q_f32(tmp + 32, c04);
  vst1q_f32(tmp + 36, c44);
  vst1q_f32(tmp + 40, c05);
  vst1q_f32(tmp + 44, c45);
  vst1q_f32(tmp + 48, c06);
  vst1q_f32(tmp + 52, c46);
  vst1q_f32(tmp + 56, c07);
  vst1q_f32(tmp + 60, c47);
  // 拷贝回矩阵C或缓冲区
  if (C_direct == 0){
    for (int j = 0; j < KERNEL_SIZE_COL; j++)
      for (int i = 0; i < KERNEL_SIZE_ROW; i++)
        CorCBuffer[j*row_CorCBuffer + i] = 0.0;
  }
  for (int j = 0; j < KERNEL_SIZE_COL; j++)
    for (int i = 0; i < KERNEL_SIZE_ROW; i++)
      CorCBuffer[j*row_CorCBuffer + i] += tmp[j*KERNEL_SIZE_ROW + i];
}

static void subblock_sgemm(int n, int REAL_P_BLOCK_SIZE, int REAL_J_BLOCK_SIZE, \
                           int REAL_I_BLOCK_SIZE, float * C) {
  for (int subj = 0; subj < REAL_J_BLOCK_SIZE; subj += KERNEL_SIZE_COL) {
    int subj_block_size = min(KERNEL_SIZE_COL, REAL_J_BLOCK_SIZE - subj);

    for (int subp = 0; subp < REAL_P_BLOCK_SIZE; subp += KERNEL_SIZE_ROW) {
      int subp_block_size = min(KERNEL_SIZE_ROW, REAL_P_BLOCK_SIZE - subp);

      float * const restrict C_ptr = C + subj*n + subp;
      if (subp_block_size==KERNEL_SIZE_ROW && subj_block_size==KERNEL_SIZE_COL)
        sgemm_kernel(REAL_I_BLOCK_SIZE, A_local + subp*REAL_I_BLOCK_SIZE, \
                     B_local + subj*REAL_I_BLOCK_SIZE, C_ptr, 1, n);
      else{
        sgemm_kernel(REAL_I_BLOCK_SIZE, A_local + subp*REAL_I_BLOCK_SIZE, \
                     B_local + subj*REAL_I_BLOCK_SIZE, C_buffer, 0, KERNEL_SIZE_ROW);
        for (int j = 0; j < subj_block_size; j++)
          for (int i = 0; i < subp_block_size; i++)
            C_ptr[n*j + i] += C_buffer[j*KERNEL_SIZE_ROW + i];
      }
    }
  }
}

void square_sgemm(int n, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  for (int j_block = 0; j_block < n; j_block += SET_J_BLOCK_SIZE){//对于B而言的水平划分
    int REAL_J_BLOCK_SIZE = min(SET_J_BLOCK_SIZE, n - j_block);

    for (int i_block = 0; i_block < n; i_block += SET_I_BLOCK_SIZE){//对于B而言的垂直划分
      int REAL_I_BLOCK_SIZE = min(SET_I_BLOCK_SIZE, n - i_block);
      // 拷贝并转置B的子块到local_B
      copy_transpose_B_into_IxJ(n, REAL_I_BLOCK_SIZE, REAL_J_BLOCK_SIZE,\
                                B + j_block*n + i_block, B_local);
      for (int p_block = 0; p_block < n; p_block += SET_P_BLOCK_SIZE) {
        int REAL_P_BLOCK_SIZE = min(SET_P_BLOCK_SIZE, n - p_block);
        // 拷贝A的子块到local_A
        copy_A_into_PxI(n, REAL_P_BLOCK_SIZE, REAL_I_BLOCK_SIZE, A + i_block*n + p_block, A_local);
        // 子块的乘法
        subblock_sgemm(n, REAL_P_BLOCK_SIZE, REAL_J_BLOCK_SIZE, REAL_I_BLOCK_SIZE, &C(p_block, j_block));
      }
    }
  }
}