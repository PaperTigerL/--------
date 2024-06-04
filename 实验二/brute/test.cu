#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void matrixMul(int *a, int *b, int *c, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < m && col < m) {
        for (int i = 0; i < m; i++) {
            sum += a[row * m + i] * b[i * m + col];
        }
        c[row * m + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <m> <n> <block_size>\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int block_size = atoi(argv[3]);
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = m * m * sizeof(int);
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            a[i * m + j] = rand() % 10;
            b[i * m + j] = rand() % 10;
        }
    }
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", c[i * m + j]);
        }
        printf("\n");
    }
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}