#include <arm_neon.h>

#define A(i,j) A[ (j)*n + (i) ]
#define B(i,j) B[ (j)*n + (i) ]
#define C(i,j) C[ (j)*n + (i) ]

const char* sgemm_desc = "Naive, three-loop sgemm.";

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 注意是列优先！
 * On exit, A and B maintain their input values. */    
// void square_sgemm (int n, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C)
void square_sgemm (int n, float* A, float* B, float* C)
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

  // 循环变换 + intrinsics
  // for (int j = 0; j < n; j++)//for each colum j of B
  //   for (int i = 0; i < n; i++){//for each row i of B
  //     register float32_t b = B[j*n+i];
  //     int p, max_4p = n & (~3);
  //     for (p = 0; p < max_4p; p+=4) {
  //       float32x4_t a = vld1q_f32(A + j*n + p);//一次拿4个float=128bits
  //       float32x4_t res = vld1q_f32(C + j*n + p);//一次拿4个float
  //       res = vmlaq_n_f32(res, a, b);
  //       vst1q_f32(C + j*n + p, res);
  //     }
  //     for (; p < n; p++)
  //       C[j*n+p] += A[j*n+p] * b;
  //   }

  // // 循环变换
  // int j, i, p;
  // for ( j = 0; j < ((n)&(~3)); j+=4)//for each colum j of B
  //   for ( i = 0; i < n; i++){//for each row i of B
  //     register float b0 = B(i,j);
  //     register float b1 = B(i,j+1);
  //     register float b2 = B(i,j+2);
  //     register float b3 = B(i,j+3);
  //     for ( p = 0; p < n; p++){
  //       C(p,j  ) += A(p,i  ) * b0;
  //       C(p,j+1) += A(p,i) * b1;
  //       C(p,j+2) += A(p,i) * b2;
  //       C(p,j+3) += A(p,i) * b3;
  //     }
  //   }
  // for ( ; j < n; j++)//for each remaining colum j of B
  //   for ( i = 0; i < n; i++){//for each row i of B
  //     register float b0 = B(i,j);
  //     for ( p = 0; p < n; p++)
  //       C(p,j  ) += A(p,i  ) * b0;
  //   }

  for (int j = 0; j < n; j++){
    for (int i = 0; i < n; i++){
      register float b = B[j*n + i];
      for (int p = 0; p < n; p++)
        C[j*n+p] += A[i*n+p] * b;
    }
  }
}
