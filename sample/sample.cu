#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

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

// CPU version
unsigned int random_u32(unsigned long long *state)
{
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state)
{ // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

// GPU version
__device__ float random_f32_device(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    unsigned int immediate = (*state * 0x2545F4914F6CDD1Dull) >> 32;
    float rand = (immediate >> 8) / 16777216.0f;
    return rand;
}

class ProbIndex
{
public:
    float prob;
    int index;
}; // struct used when sorting probabilities during top-p sampling

int compare(const void *a, const void *b)
{
    ProbIndex *a_ = (ProbIndex *)a;
    ProbIndex *b_ = (ProbIndex *)b;
    if (a_->prob > b_->prob)
        return -1;
    if (a_->prob < b_->prob)
        return 1;
    return 0;
}

__device__ int compare_device(const void* a, const void* b) {
    ProbIndex *a_ = (ProbIndex *)a;
    ProbIndex *b_ = (ProbIndex *)b;
    if (a_->prob > b_->prob) {
        return -1;
    }
    if (a_->prob < b_->prob) {
        return 1;
    }
    return 0;
}

void softmax_cpu(float *x, int size)
{
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

int sample_argmax(float *probabilities, int n)
{
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++)
    {
        if (probabilities[i] > max_p)
        {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin)
{
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++)
    {
        if (probabilities[i] >= cutoff)
        {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++)
    {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp)
        {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++)
    {
        cdf += probindex[i].prob;
        if (r < cdf)
        {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

int sample_mult(float *probabilities, int n, float coin)
{
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++)
    {
        cdf += probabilities[i];
        if (coin < cdf)
        {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int sample(float *logits, float temperature, int vocab_size, float topp, unsigned long long rng_state, ProbIndex *probindex)
{
    // sample the token given the logits and some hyperparameters
    int next;
    if (temperature == 0.0f)
    {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, vocab_size);
    }
    else
    {
        // apply the temperature to the logits
        for (int q = 0; q < vocab_size; q++)
        {
            logits[q] /= temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax_cpu(logits, vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&rng_state);
        // we sample from this distribution to get the next token
        if (topp <= 0 || topp >= 1)
        {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, vocab_size, coin);
        }
        else
        {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, vocab_size, topp, probindex, coin);
        }
    }
    return next;
}

__device__ void sampel_argmax_kernel(float *probabilities, int n, int *next) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++)
    {
        if (probabilities[i] > max_p)
        {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    *next = max_i;
}

__device__ void softmax_kernel(float *x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }    
}

__device__ void sample_topp_kernel(float *probabilities, int n, float topp, ProbIndex *probindex, float coin, int *next) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++)
    {
        if (probabilities[i] > cutoff)
        {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }

    // Sort probindex array (using thrust would be more efficient for larger arrays)
    for (int i = 0; i < n0 - 1; i++) {
        for (int j = 0; j < n0 - i - 1; j++) {
            if (compare_device(&probindex[j], &probindex[j + 1]) > 0) {
                ProbIndex temp = probindex[j];
                probindex[j] = probindex[j + 1];
                probindex[j + 1] = temp;
            }
        }
    }

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++)
    {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp)
        {
            last_idx = i;
            break;
        }
    }
    
    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i < last_idx; i++)
    {
        cdf += probindex[i].prob;
        if (r < cdf)
        {
            *next = probindex[i].index;
            return;
        }
    }
    *next = probindex[last_idx].index; // in case of rounding errors
}

__global__ static void sample_kernel(float *d_logits, float *d_coin, float temperature, int vocab_size, float topp, unsigned long long rng_state, ProbIndex *d_probindex, int *d_next)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (temperature == 0.0f) {
        sampel_argmax_kernel(d_logits, vocab_size, d_next);
    } else {
        for (int q = 0; q < vocab_size; q++) {
            d_logits[q] /= temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax_kernel(d_logits, vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32_device(&rng_state);
        // we sample from this distribution to get the next token
        if (topp <= 0 || topp >= 1) {
            // simply sample from the predicted probability distribution
            sampel_argmax_kernel(d_logits, vocab_size, d_next);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            sample_topp_kernel(d_logits, vocab_size, topp, d_probindex, coin, d_next);
        }
    }
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
    float *logits = new float[vocab_size];
    int h_next = 0;
    initializeArray(logits, vocab_size);
    ProbIndex *probindex = (ProbIndex *)malloc(vocab_size * sizeof(ProbIndex));

    // device allocation
    float *d_logits;
    int *d_next;
    float *d_coin;
    ProbIndex *d_probindex;
    CHECK_CUDA(cudaMalloc((void **)&d_logits, vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_next, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_coin, sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_probindex, vocab_size * sizeof(ProbIndex)));
    CHECK_CUDA(cudaMemcpy(d_logits, logits, vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    h_next = sample(logits, temperature, vocab_size, topp, rng_state, probindex);
    std::cout << "h_next: " << h_next << std::endl;

    dim3 gridSize(1);
    dim3 blockSize(1);
    sample_kernel<<<1, 1>>>(d_logits, d_coin, temperature, vocab_size, topp, rng_state, d_probindex, d_next);
    CHECK_CUDA(cudaGetLastError());

    int d_next_transfered;
    CHECK_CUDA(cudaMemcpy(&d_next_transfered, d_next, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "d_next_transfered: " << d_next_transfered << std::endl;

    return 0;
}
