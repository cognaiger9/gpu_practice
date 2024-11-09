CUC = nvcc
CUDA_FLAGS = -gencode arch=compute_86,code=sm_86 -lcublas # rtx

# the most basic way of building that is most likely to work on most systems
.PHONY: main
main: transpose.cu
	$(CUC) $(CUDA_FLAGS) -o main transpose.cu

max: arg_max.cu
	$(CUC) $(CUDA_FLAGS) -o max arg_max.cu

.PHONY: clean
clean:
	rm -f main