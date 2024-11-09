#include <cstdio>
#include <cuda.h>

#include "matmul.h"
#include "util.h"

void naive_cpu_matmul(half* xout, half* x, half* _C, int n, int d, int batch)
{
  // W(d, n) @ x(batch, n) -> xout(batch, d)
  for (int b = 0; b < batch; b++)
  {
    for (int i = 0; i < d; i++)
    {
      for (int j = 0; j < n; j++)
      {
        xout[b * d + i] += x[b * n + j] * _C[i * n + j];
      }
    }
  }
}

void matmul(half* xout, half* x, half* w, int n, int d, int batch, cublasHandle_t handle)
{
  // naive_cpu_matmul(_A, _B, _C, M, N, K);
  // W(d, n) @ x(batch, n) -> xout(batch, d)
  const __half alpha = static_cast<__half>(1), beta = static_cast<__half>(0);
  CHECK_CUBLAS(cublasHgemm(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           d,
                           batch,
                           n,
                           &alpha,
                           w,
                           d,
                           x,
                           n,
                           &beta,
                           xout,
                           d));
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K)
{
}

//void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K)
//{
//}