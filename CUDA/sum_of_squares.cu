#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 2048
#define THREADS_PER_BLOCK 1024

__global__ void kernel(int *a, int *b, unsigned *c) {
  __shared__ int temp[THREADS_PER_BLOCK];

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  temp[threadIdx.x] = a[idx] * b[idx];

  __syncthreads();

  if (threadIdx.x == 0) {
    unsigned sum = 0;
    for (int i = 0; i < THREADS_PER_BLOCK; i++) {
      sum += temp[i];
    }
    atomicAdd(c, sum);
  }
}

int main() {
  int *a, *b;
  int *d_a, *d_b;
  unsigned *d_c;
  int malloc_size = sizeof(int) * SIZE;
  unsigned total_sum;

  a = (int *)malloc(malloc_size);
  b = (int *)malloc(malloc_size);

  cudaMalloc(&d_a, malloc_size);
  cudaMalloc(&d_b, malloc_size);
  cudaMalloc(&d_c, sizeof(unsigned));

  for (int i = 0; i < SIZE; i++) {
    a[i] = b[i] = i + 1;
  }

  cudaMemcpy(d_a, a, malloc_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, malloc_size, cudaMemcpyHostToDevice);

  kernel<<<SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

  cudaMemcpy(&total_sum, d_c, sizeof(unsigned), cudaMemcpyDeviceToHost);

  printf("Total Sum: %u\n", total_sum);

  free(a);
  free(b);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}