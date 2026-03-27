# sgemm-serial
An single-precision dense matrix multiplication optimization for Arm architecture (Kunpeng920)

For detailed implementation, please refer to my blog: https://blog.csdn.net/weixin_43614211/article/details/122105195

# 以下为《稠密矩阵乘法单核优化：Arm》博客的内容
（由于上述的CSDN需要付费才能看到完整博客内容，违背本人技术分享的初衷，此处附载博客全文）

本次作业为优化单核上的稠密矩阵乘法。
以下各个优化方法的效果均是在它之前叠加的。

机器：鲲鹏920芯片，只用64个核中的一个
提供的矩阵输入为列主序。

## 编译选项
对于naïve版本的代码，如下所示，不妨先“无脑”地加上`-O3 -fomit-frame-pointer -march=armv8-a -ffast-math`等编译选项来让编译器尽可能提供些自动向量化的效果。
```c
void square_sgemm (int n, float* A, float* B, float* C) {
  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j)  {
      /* Compute C(i,j) */
      float cij = C[i+j*n];
      for( int k = 0; k < n; k++ )
        cij += A[i+k*n] * B[k+j*n];
      C[i+j*n] = cij;
  }
}
```
仅仅是如此，在不同规模的算例上性能就已经有2~10倍的提升，如下图所示。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e89c3d6accdb1120dfa966263dfb1551.png#pic_center)

可以看到n每逢4的倍数便有显著的性能下降，这是cache thrashing导致的。可做半定量分析：课程集群L1 cache为64B/line，4路组相联，256个组，可知地址低6位为Offset，中间8位为Index，高位为Tag。N-way set associativity只是提供了conflict miss时的“容错性”，因此不失一般性，假定为direct-mapped来分析。地址每隔2^14B就会拥有相同的Index而被映射到同一个set上，对于单精度浮点数而言就是4096个数，因此当n满足(n*m)%4096==0时（m=1,2,…,n-1），就会在一轮k维的循环中产生cache conflict miss，m就是冲突发生时两个B元素相隔的行数。因此冲突频率随n增大而增大，当n≥4096时，就是每两次相邻的对B元素读取都会造成冲突。

## 循环变换
注意到在naïve的代码中，由于矩阵采用列主序的存储方式，因此先行后列的方式来计算C中元素的值，虽然对B元素访存是连续的，但对于C和A矩阵的访存都是不利的。尤其在循环最内维的k维，A[i+k*n]是大跨步跳跃式访存。
因此可以采用对i和j维的循环交换，来发掘数据复用的空间局部性。代码如下所示。
```c
void square_sgemm (int n, float* A, float* B, float* C) {
  for (int j = 0; j < n; j++){
    for (int i = 0; i < n; i++){
      register float b = B[j*n + i];
      for (int p = 0; p < n; p++)
        C[j*n+p] += A[i*n+p] * b;
    }
  }
}

```
示意图如下图，相当于按列主序遍历B中元素，对于其中的每个元素b，找到它对应有贡献的C和A中的元素所在的列，进行乘加计算。最内维的p维循环对A和C都是连续的，可以有效利用向量化。由于更改循环后，在整轮最内维的p循环中，b的元素是固定不变的寄存器变量，因此不再出现步骤一中的cache conflict miss，反而是矩阵规模n每逢4的倍数就比相邻的有提升，这是因为n为4的倍数能刚好被向量化指令覆盖，而不会多出额外的数据需要标量运算。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/07ffdee61848e1c370f7f364186dceb9.png#pic_center)

## 消除指针别名
消除指针别名告诉编译器修改指针指向的内存内容只能经过该指针之手，使编译器有更大优化空间。主要方法是给函数形参中的指针添加`__restrict__`关键字。其它局部的指针变量在定义时也可用此修饰。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6627d7876000ea82a2efdc9656dc22fc.png#pic_center)

循环变换和消除别名的性能均有明显提升，如下图所示。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9280ab60bebef81364e8c5450b646279.png#pic_center)

## 循环展开
将循环展开，同时做多列的乘加操作，即取同行不同列的B矩阵元素b0, b1, b2, b3，均与相同的A列做乘法后加到不同的C列上。代码如下所示，需要注意处理余下不足4的列。。
```c
int j, i, p;
for ( j = 0; j < ((n)&(~3)); j+=4)//for each colum j of B
  for ( i = 0; i < n; i++){//for each row i of B
    register float b0 = B(i,j);
    register float b1 = B(i,j+1);
    register float b2 = B(i,j+2);
    register float b3 = B(i,j+3);
    for ( p = 0; p < n; p++){
      C(p,j  ) += A(p,i) * b0;
      C(p,j+1) += A(p,i) * b1;
      C(p,j+2) += A(p,i) * b2;
      C(p,j+3) += A(p,i) * b3;
    }
  }
for ( ; j < n; j++)//for each remaining colum j of B
  for ( i = 0; i < n; i++){//for each row i of B
    register float b0 = B(i,j);
    for ( p = 0; p < n; p++)
      C(p,j  ) += A(p,i  ) * b0;
  }
```
实验效果显示选4列为一批做乘加效果较好，而大于4列则效果开始下降。**循环展开常见的是对最内层做**，优势在于循环开销（如终止条件上的分支和计数器变量的更新）的减少。至于为什么要在最外层循环做展开（而不是最内层循环），需要从访存优化的角度来看。对比上一节《循环变换》中最内层循环只有一句`C[j*n+p] += A[i*n+p] * b;`，展开后此处最内层循环有四句`C(p,j  ) += A(p,i) * b0;`。注意，改写后，`A(p,i)`只需要载入寄存器一次，就能服务于`C(p,j  )`，`C(p,j+1)`，`C(p,j+2)`，`C(p,j+3)`等的计算；而原来，相同的`A[i*n+p]`值需要为每个`C[j*n+p]`加载一次。因此，外层循环的展开将矩阵`A`元素加载次数减少了nb倍（nb为循环展开的项数，这里是4）。更详细的关于外层循环展开的分析（如中、外层同时展开），可以参见这篇[技术资料](https://techpubs.jurassic.nl/manuals/0640/developer/OrOn2_PfTune/sgi_html/ch07.html#id20090)。当然，按这么分析，outermost loop unrolling这种访存优化对于stencil计算是起不到效果的。

在[有的书里](https://www.sciencedirect.com/topics/computer-science/loop-unrolling)，也有类似于这样外层循环展开的写法（只有一层循环，但分成多part，循环体内对每part都做一次计算），分析认为是可以更有效地利用执行单元的流水特性，从而在相同的cycle数内完成更多的计算。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/42af945b1b4f77b979d08c85b59cfab7.png#pic_center)

依然存在n不能整除4时性能下降的情况，因为需要额外执行一段代码。

## 内存对齐和简单Blocking
利用分块技术提高计算访存比获得更高的性能是常用的优化手段。从之前的代码来看，有三层循环（从外到内依次是j -> i -> p），因此可以在这3个维度上采取分块，分别设为`SET_J_BLOCK_SIZE`, `SET_I_BLOCK_SIZE`, `SET_P_BLOCK_SIZE`。越内维访存越连续，因此设的分块大小更大。此处同时配合内存对齐的手段，是因为对于每一个分块矩阵的乘法，单独将A和B拷贝到一块对齐的连续的内存A_local和B_local中，计算结果存到同样对齐的连续的C_local中。一个好处是A_local和B_local矩阵在拷贝时已经预热，放进了CPU的cache里；另一个好处是在真正计算时，读取和存储都是连续的，提高了cache效率。将一块`realM`x`realN`大小的矩阵拷贝到`setM`x`setN`大小的内存中的代码如下所示。
```c
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
```
整体的计算逻辑如下所示，仅做了一级分块，其中计算部分类似前面步骤四中的j以步长4为单位做循环。区别在于分块后为减低寻址开销，每个分块用局部的指针	`xxx_local_ptr`指示当前计算的位置。拷贝分块矩阵的函数`copy_into_MxN_nopadding`与步骤六中的函数`copy_PxI_nopadding()`几乎一样。为了寻找这组最优的分块，可以通过编一个简单的Shell脚本，设置环境变量来指定各维度的分块，然后在Makefile里根据环境变量定义宏，再编译和运行。

```c
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
```
在有限的实验中（时间所限并没有遍历整个搜索空间），`SET_J_BLOCK_SIZE`=96，`SET_I_BLOCK_SIZE`=100和`SET_P_BLOCK_SIZE`=124的分块效果较好，性能结果如下图所示。可见相比之前有了接近2倍的提升，大部分维持在17~18 GFlop/s的水平。可以看到应用分块技术后，因矩阵规模而导致的性能波动减小了。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0eaaee934eab44ed3d8a5bf0d2f5800b.png#pic_center)

# 两级Blocking+转置重组
为了更细致的优化，可以做二级分块，在原有基础上，在拷贝出来的对齐且连续的A_local和B_local内做进一步的分块，每次计算一个`KERNEL_SIZE_ROW` x `KERNEL_SIZE_COL`大小的矩阵乘法。需要说明的是，此部分二级分块的内容参考了[Github上的代码](https://github.com/xiaoyi-jason/simple_gemm)，改写融入到原有一级分块的框架中，故此优化想法并非完全出自本人。由于使用了arm neon的intrinsics，每次一次性对A_local和C_local内的4个浮点数操作，故在此处拷贝A和B时使用padding 0来补齐原矩阵分块无法填满`A_local`和`B_local`的地方。下图在一级分块中调用二级分块的矩阵乘法`subblock_sgemm()`函数。类似地，下图的二级分块的乘法调用最内核的`sgemm_kernel()`完成固定大小的`KERNEL_SIZE`的小矩阵乘法。此处设置`KERNEL_SIZE_ROW` = `KERNEL_SIZE_COL`=8.

```c
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
```
在内核函数`sgemm_kernel`中，利用CPU提供的128bits定长寄存器，通过intrinsics指令完成SIMD操作。基本逻辑是

 1. 从小分块内存加载到定长寄存器
 2. 乘加操作得到结果
 3. 结果从寄存器存储回小分块内存
 4. 拷回C矩阵或为补齐而设的缓冲区中

```c
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
```
值得一提的是，原作者在这里拷贝A和B矩阵时，使元素位置重组，设计得很精妙，使得后续计算时对`B_local`的访存与`A_local`保持一致的pattern，连续高效。这部分较为难懂，按个人理解，计算逻辑的**示意图**如下。下图中`setX`即为上文提到的`SET_X_BLOCK_SIZE`，`realX`即为`REAL_X_BLOCK_SIZE`，而`KernelRow`和`KernelCol`分别为`KERNEL_SIZE_ROW`和`KERNEL_SIZE_COL`。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d8f86b980d7321b4a35f940871f71a02.png#pic_center)

拷贝并重组存储顺序的代码如下。

```c
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
```

最后的性能优化结果如下图所示，图中包括了**基准OpenBLAS**的结果。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bfb90e73ef797aa1b2bc2e69b702cdde.png#pic_center)
