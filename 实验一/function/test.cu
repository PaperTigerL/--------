#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void pi_calculate(double *pi, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double sum = 0.0;
        for (int i = idx; i < m; i += n) {
            sum += (i % 2 == 0 ? 1 : -1) * 4.0 / (2 * i + 1);
        }
        pi[idx] = sum;
    }
}

double calculate_pi(int n, int m) {
    double *pi, *d_pi;
    pi = (double *) malloc(n * sizeof(double));
    cudaMalloc(&d_pi, n * sizeof(double));
    pi_calculate<<<(n + 255) / 256, 256>>>(d_pi, n, m);
    cudaMemcpy(pi, d_pi, n * sizeof(double), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += pi[i];
    }
    cudaFree(d_pi);
    free(pi);
    return sum ;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <n> <m>\n", argv[0]);
        return 0;
    }
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    double result = calculate_pi(n, m);
printf("The calculated value of Pi is: %.10lf\n", result); 
    return 0;
}

