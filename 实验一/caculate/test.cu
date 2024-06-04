#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void estimatePI(float *result, int num_samples, float step)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (tid + 0.5f) * step;  // Calculate x for this thread

    if (x < 1.0f)
        atomicAdd(result, sqrt(1.0f - x * x) * step);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <num_intervals> <num_threads>" << std::endl;
        return 1;
    }

    int num_intervals = std::atoi(argv[1]);  // Number of intervals for integration
    int num_threads = std::atoi(argv[2]);    // Number of threads
    float step = 1.0f / static_cast<float>(num_intervals);  // Step size
    int num_blocks = (num_intervals + num_threads - 1) / num_threads;
    float *d_result;
    float h_result = 0.0f;

    // Allocate device memory for the result
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    estimatePI<<<num_blocks, num_threads>>>(d_result, num_intervals, step);

    // Copy the result back to the host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Multiply by 4 to get the estimate of PI using integration
    float pi_estimate = 4.0f * h_result;

    std::cout << "Estimated PI using integration: " << pi_estimate << std::endl;

    // Clean up
    cudaFree(d_result);

    return 0;
}