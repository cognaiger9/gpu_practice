#pragma once
#include "util.h"

void naive_cpu_matmul(half* xout, half* x, half* _C, int n, int d, int batch);
void matmul(half* xout, half* x, half* w, int n, int d, int batch, cublasHandle_t handle);
void matmul_init(int M, int N, int K);
//void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K);