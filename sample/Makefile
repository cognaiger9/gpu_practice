CUC = nvcc
CPP = g++
CUDA_FLAGS = -gencode arch=compute_86,code=sm_86 # rtx

# the most basic way of building that is most likely to work on most systems
.PHONY: main
main: sample.cu
	$(CUC) $(CUDA_FLAGS) -o main sample.cu

.PHONY: clean
clean:
	rm -f main
