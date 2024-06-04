#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void matrixPow(int *a, int *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < m && col < m) {
        if (n == 0) {
            c[row * m + col] = (row == col) ? 1 : 0;
        } else if (n == 1) {
            c[row * m + col] = a[row * m + col];
        } else {
            int* temp = new int[m];
            for (int i = 0; i < m; i++) {
                temp[i] = a[row * m + i];
            }
            for (int k = 2; k <= n; k++) {
                int* temp2 = new int[m];
                for (int i = 0; i < m; i++) {
                    int temp_sum = 0;
                    for (int j = 0; j < m; j++) {
                        temp_sum += temp[j] * a[j * m + i];
                    }
                    temp2[i] = temp_sum;
                }
                delete[] temp;
                temp = temp2;
            }
            c[row * m + col] = temp[col];
            delete[] temp;
        }
    }
}
int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <m> <n> <block_dim_x> <block_dim_y>\n", argv[0]);
        return 1;
    }
    int m = 4;
    int n_values[] = {2, 3, 4};
    int block_values[] = {8, 16, 32};

    int *a, *c;
    int *d_a, *d_c;
    int size = m * m * sizeof(int);
    a = (int*)malloc(size);
    c = (int*)malloc(size);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_c, size);
        for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            a[i * m + j] = rand() % 10;
        }
    }
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < sizeof(n_values) / sizeof(n_values[0]); i++) {
        int n = n_values[i];
        for (int j = 0; j < sizeof(block_values) / sizeof(block_values[0]); j++) {
            int block = block_values[j];

            dim3 block_dim(block, block);
            dim3 grid_dim((m + block - 1) / block, (m + block - 1) / block);

            matrixPow<<<grid_dim, block_dim>>>(d_a, d_c, m, n);

            cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

            printf("Results for n=%d, block_dim=%d:\n", n, block);
            for (int row = 0; row < m; row++) {
                for (int col = 0; col < m; col++) {
                    printf("%d ", c[row * m + col]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    free(a);
    free(c);
    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}