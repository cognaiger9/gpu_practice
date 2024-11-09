#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CHECK_CUBLAS(call)                                                        \
  do                                                                              \
  {                                                                               \
    cublasStatus_t status_ = call;                                                \
    if (status_ != CUBLAS_STATUS_SUCCESS)                                         \
    {                                                                             \
      fprintf(stderr, "CUBLAS error (%s:%d): %d\n", __FILE__, __LINE__, status_); \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (0)

double get_time();

//void check_matmul(float *A, float *B, float *C, int M, int N, int K);

void print_mat(float *m, int R, int C);

//float* alloc_mat(int R, int C);
half* alloc_mat_half(int R, int C);

void rand_mat(half* m, int R, int C);

void zero_mat(half *m, int R, int C);