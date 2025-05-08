#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    
    {                                                                       
        cudaError_t err = call;                                             
        if (err != cudaSuccess) {                                           
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
            return -1;                                                      
        }                                                                   
    }

__global__ void addVectors(int* A, int* B, int* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int n = 1000000;
    int* A, * B, * C;
    int size = n * sizeof(int);

    CHECK_CUDA(cudaMallocHost(&A, size));
    CHECK_CUDA(cudaMallocHost(&B, size));
    CHECK_CUDA(cudaMallocHost(&C, size));

    for (int i = 0; i < n; i++)
    {
        A[i] = i;
        B[i] = i * 2;
    }

    int* dev_A, * dev_B, * dev_C;
    CHECK_CUDA(cudaMalloc(&dev_A, size));
    CHECK_CUDA(cudaMalloc(&dev_B, size));
    CHECK_CUDA(cudaMalloc(&dev_C, size));

    CHECK_CUDA(cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);

    CHECK_CUDA(cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost));

    // Output result
    std::cout << "First 10 results:\n";
    for (int i = 0; i < 10; i++)
    {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    CHECK_CUDA(cudaFree(dev_A));
    CHECK_CUDA(cudaFree(dev_B));
    CHECK_CUDA(cudaFree(dev_C));
    CHECK_CUDA(cudaFreeHost(A));
    CHECK_CUDA(cudaFreeHost(B));
    CHECK_CUDA(cudaFreeHost(C));

    return 0;
}

