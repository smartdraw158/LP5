#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul(int* A, int* B, int* C, int N) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < N && Col < N) {
        int Pvalue = 0;
        for (int k = 0; k < N; k++) {
            Pvalue += A[Row * N + k] * B[k * N + Col];
        }
        C[Row * N + Col] = Pvalue;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N = 512;
    int size = N * N * sizeof(int);

    int* A, * B, * C;
    int* dev_A, * dev_B, * dev_C;

    std::cout << "Allocating host memory..." << std::endl;
    checkCudaError(cudaMallocHost(&A, size), "cudaMallocHost A");
    checkCudaError(cudaMallocHost(&B, size), "cudaMallocHost B");
    checkCudaError(cudaMallocHost(&C, size), "cudaMallocHost C");

    std::cout << "Allocating device memory..." << std::endl;
    checkCudaError(cudaMalloc(&dev_A, size), "cudaMalloc dev_A");
    checkCudaError(cudaMalloc(&dev_B, size), "cudaMalloc dev_B");
    checkCudaError(cudaMalloc(&dev_C, size), "cudaMalloc dev_C");

    std::cout << "Initializing matrices..." << std::endl;
    for (int i = 0; i < N * N; ++i) {
        A[i] = i;
        B[i] = i;
    }

    std::cout << "Copying data to device..." << std::endl;
    checkCudaError(cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCudaError(cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice), "cudaMemcpy B");

    std::cout << "Launching kernel..." << std::endl;
    dim3 dimBlock(16, 16);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::cout << "Copying result back to host..." << std::endl;
    checkCudaError(cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    std::cout << "Top-left 10x10 block of matrix C:" << std::endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}

