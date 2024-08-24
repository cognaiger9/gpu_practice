#include <cub/cub.cuh>
#include <cuda_runtime.h>
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

//
// Block-sorting CUDA kernel
//
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(int *d_in, int *d_out)
{
  // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
  using BlockLoadT = cub::BlockLoad<
      int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<
      int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE>;
  using BlockRadixSortT = cub::BlockRadixSort<
      int, BLOCK_THREADS, ITEMS_PER_THREAD>;

  // Allocate type-safe, repurposable shared memory for collectives
  __shared__ union
  {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockRadixSortT::TempStorage sort;
  } temp_storage;

  // Obtain this block's segment of consecutive keys (blocked across threads)
  int thread_keys[ITEMS_PER_THREAD];
  int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
  BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);

  __syncthreads(); // Barrier for smem reuse

  // Collectively sort the keys
  BlockRadixSortT(temp_storage.sort).Sort(thread_keys);

  __syncthreads(); // Barrier for smem reuse

  // Store the sorted segment
  BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
  __syncthreads();
}

// Function to generate random for array
void initializeArray(int *d_in, int size)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 1000);

  for (int i = 0; i < size; i++)
  {
    d_in[i] = dis(gen);
  }
}

// Verify output from gpu
bool verifyOutput(int *h_in, int *h_out, int size, int BLOCK_THREADS, int ITEMS_PER_THREAD)
{
  // Sort the host output
  int step = BLOCK_THREADS * ITEMS_PER_THREAD;
  for (int i = 0; i < size; i += step)
  {
    std::sort(h_in + i, h_in + std::min(i + step, size));
  }

  // Compare the sorted host output with the sorted device output
  for (int i = 0; i < size; i++)
  {
    if (h_in[i] != h_out[i])
    {
      printf("sorted_cpu[%d] = %d, d_out[%d] = %d\n", i, h_in[i], i, h_out[i]);
      return false;
    }
  }

  return true;
}

int main()
{
  const int BLOCK_THREADS = 128;
  const int ITEM_PER_THREAD = 16;
  const int ITEM_PER_BLOCK = BLOCK_THREADS * ITEM_PER_THREAD;
  const int NUM_BLOCKS = 32;
  const int TOTAL_ITEM = NUM_BLOCKS * ITEM_PER_BLOCK;

  // Allocate host memory
  int *h_in = new int[TOTAL_ITEM];
  int *h_out = new int[TOTAL_ITEM];

  // Allocate device memory
  int *d_in;
  int *d_out;
  CHECK_CUDA(cudaMalloc((void **)&d_in, TOTAL_ITEM * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_out, TOTAL_ITEM * sizeof(int)));

  initializeArray(h_in, TOTAL_ITEM);
  CHECK_CUDA(cudaMemcpy(d_in, h_in, TOTAL_ITEM * sizeof(int), cudaMemcpyHostToDevice));

  BlockSortKernel<BLOCK_THREADS, ITEM_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(d_in, d_out);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(h_out, d_out, TOTAL_ITEM * sizeof(int), cudaMemcpyDeviceToHost));

  if (verifyOutput(h_in, h_out, TOTAL_ITEM, BLOCK_THREADS, ITEM_PER_THREAD))
  {
    std::cout << "Success!" << std::endl;
  }
  else
  {
    std::cout << "Failed!" << std::endl;
  }
}