/*
Compile:
    nvcc benchmark_pinned_vs_unified.cu -o bench
Run:
    ./bench

This code times a simple vector add with both pinned (page-locked) host memory
and unified memory. For pinned memory, we explicitly do host->device and device->host
copies. For unified memory, we directly access the same pointers from CPU and GPU.

If you don't see a meaningful difference with n=1<<25 (which is ~128MB of floats),
try increasing n to 1<<26 or 1<<27.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

float runPinnedVectorAdd(int n) {
    // Timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t size = n * sizeof(float);

    // Allocate host pinned memory
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);

    // Init
    for(int i = 0; i < n; i++){
        h_A[i] = 1.f; 
        h_B[i] = 2.f;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Start timer
    cudaEventRecord(start);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;
    vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // Copy back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float runUnifiedVectorAdd(int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t size = n * sizeof(float);

    // Unified memory
    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Init
    for(int i = 0; i < n; i++){
        A[i] = 1.f;
        B[i] = 2.f;
    }

    // Start timer
    cudaEventRecord(start);

    // Kernel launch
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;
    vecAdd<<<gridSize, blockSize>>>(A, B, C, n);

    // Sync
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    // Access results on CPU (already available)
    // (Optionally verify one element)
    // printf("C[0] = %f\n", C[0]);

    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    // For a real difference, try 1<<25 (~128MB) or higher.
    const int n = 1 << 25;

    float timePinned   = runPinnedVectorAdd(n);
    float timeUnified  = runUnifiedVectorAdd(n);

    printf("Pinned   : %f ms\n", timePinned);
    printf("Unified  : %f ms\n", timeUnified);
    return 0;
}
