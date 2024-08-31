#include <random>
#include "cublas_v2.h"

#define CHECK_CUDA(call)                                                 \
  do                                                                     \
  {                                                                      \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess)                                          \
    {                                                                    \
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

// Function to generate random for array
void initializeArray(float *h_in, int size)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 10.0f);

  for (int i = 0; i < size; i++)
  {
    h_in[i] = dis(gen);
  }
}

float* transpose(float *h_in, int row, int col) {
    float* transposed = new float[row * col];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            transposed[j * row + i] = h_in[i * col + j];
        }
    }
    return transposed;
}

void transpose_gpu(float* d_in, float* d_out, int row, int col) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float zero = 0.0f;
    float one = 1.0f;
    CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, row, col, &one, d_in, col, &zero, d_in, row, d_out, row));

    CHECK_CUBLAS(cublasDestroy(handle));
}

void verify_output(float* h_out, float* d_out, int row, int col) {
    for (int i = 0; i < row * col; i++) {
        if (h_out[i] != d_out[i]) {
            printf("h_out[%d] = %f, d_out[%d] = %f\n", i, h_out[i], i, d_out[i]);
            return;
        }
    }
    printf("Successful");
}

int main() {
    const int row = 555, col = 333;
    float *h_in = new float[row * col];
    initializeArray(h_in, row * col);
    float *h_out = transpose(h_in, row, col);

    float *d_in;
    float *d_out;
    float *d_out_host = new float[row * col];
    CHECK_CUDA(cudaMalloc((void **)&d_in, row * col * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_out, row * col * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, row * col * sizeof(float), cudaMemcpyHostToDevice));

    transpose_gpu(d_in, d_out, row, col);

    CHECK_CUDA(cudaMemcpy(d_out_host, d_out, row * col * sizeof(float), cudaMemcpyDeviceToHost));
    verify_output(h_out, d_out_host, row, col);
    return 0;
}