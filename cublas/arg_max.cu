#include <iostream>
#include <random>
#include <vector>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t status_ = call;                                            \
        if (status_ != cudaSuccess)                                            \
        {                                                                      \
            fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
                    cudaGetErrorName(status_), cudaGetErrorString(status_));   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                              \
    do                                                                                  \
    {                                                                                   \
        cublasStatus_t status_ = call;                                                  \
        if (status_ != CUBLAS_STATUS_SUCCESS)                                           \
        {                                                                               \
            fprintf(stderr, "CUBLAS error (%s:%d): %d\n", __FILE__, __LINE__, status_); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

int argmax(float *prob, int n)
{
    int max_i = 0;
    float max_p = prob[0];
    for (int i = 1; i < n; i++)
    {
        if (prob[i] > max_p)
        {
            max_i = i;
            max_p = prob[i];
        }
    }
    return max_i;
}

int argmax_cublas(float *prob, int n)
{
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int res;
    CHECK_CUBLAS(cublasIsamax(handle, n, prob, 1, &res));

    CHECK_CUBLAS(cublasDestroy(handle));
    return res;
}

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

int main()
{
    float temperature = 1.0f;
    float topp = 0.9f;
    unsigned long long rng_state = 314028;
    int vocab_size = 10000;

    // host allocation
    float *h_in = new float[vocab_size];
    int h_next = 0;
    initializeArray(h_in, vocab_size);
    h_next = argmax(h_in, vocab_size);
    std::cout << "h_next: " << h_next << std::endl;

    // device allocation
    float *d_logits;
    CHECK_CUDA(cudaMalloc((void **)&d_logits, vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_logits, h_in, vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    int d_next = argmax_cublas(d_logits, vocab_size);
    std::cout << "d_next: " << d_next << std::endl;

    return 0;
}
