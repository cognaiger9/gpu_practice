#include <cub/cub.cuh>
#include <random>
#include <vector>

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
    __shared__ union {
        typename BlockLoadT::TempStorage       load;
        typename BlockStoreT::TempStorage      store;
        typename BlockRadixSortT::TempStorage  sort;
    } temp_storage;

    // Obtain this block's segment of consecutive keys (blocked across threads)
    int thread_keys[ITEMS_PER_THREAD];
    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);

    __syncthreads();        // Barrier for smem reuse

    // Collectively sort the keys
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);

    __syncthreads();        // Barrier for smem reuse

    // Store the sorted segment
    BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
}

// Function to generate random for array
void initializeArray(int *d_in, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000);

    for (int i = 0; i < size; i++) {
        d_in[i] = dis(gen);
    }
}

// Verify output from gpu
bool verifyOutput(int *d_in, int *d_out, int size) {
    std::vector<int> sorted_cpu(size);
    std::sort(sorted_cpu.begin(), sorted_cpu.end());

    // Compare the sorted host output with the sorted device output
    for (int i = 0; i < size; i++) {
        if (sorted_cpu[i] != d_out[i]) {
            return false;
        }
    }
    return true;
}

int main() {
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
    cudaMalloc((void **) &d_in, TOTAL_ITEM * sizeof(int));
    cudaMalloc((void **) &d_out, TOTAL_ITEM * sizeof(int));

    initializeArray(h_in, TOTAL_ITEM);
    cudaMemcpy(d_in, h_in, TOTAL_ITEM * sizeof(int), cudaMemcpyHostToDevice);

    BlockSortKernel<BLOCK_THREADS, ITEM_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, TOTAL_ITEM * sizeof(int), cudaMemcpyDeviceToHost);

    if (verifyOutput(h_in, h_out, TOTAL_ITEM)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failed!" << std::endl;
    }
}