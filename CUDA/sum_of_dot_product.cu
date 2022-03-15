#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024
#define THREADS_PER_BLOCK 1024

__device__ double getValue(int row, int col, double *matrix_list) {
  return matrix_list[row * SIZE + col];
}

__device__ int getRowInd(int idx) { return (int)(idx / SIZE); }

__device__ int getColInd(int idx) { return (idx % SIZE); }

__device__ void getMulti(int idx, double *A, double *B, double *C) {
  C[idx] = 0.;

  int row = getRowInd(idx);
  int col = getColInd(idx);

  for (int i = 0; i < SIZE; i++) {
    C[idx] += getValue(row, i, A) * getValue(i, col, B);
  }
}

__global__ void kernel(double *A, double *B, double *C) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < SIZE * SIZE) {
    getMulti(idx, A, B, C);
  }
}

int main() {
  double *A, *B, *C;
  double *d_A, *d_B, *d_C;
  double total;
  cudaEvent_t d_begin, d_end;
  float elapsedTime;

  A = (double *)malloc(sizeof(double) * SIZE * SIZE);
  B = (double *)malloc(sizeof(double) * SIZE * SIZE);
  C = (double *)malloc(sizeof(double) * SIZE * SIZE);

  cudaMalloc(&d_A, sizeof(double) * SIZE * SIZE);
  cudaMalloc(&d_B, sizeof(double) * SIZE * SIZE);
  cudaMalloc(&d_C, sizeof(double) * SIZE * SIZE);

  for (int i = 0; i < SIZE * SIZE; i++) {
    A[i] = B[i] = 1.;
    C[i] = 0.;
  }

  cudaMemcpy(d_A, A, sizeof(double) * SIZE * SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(double) * SIZE * SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(double) * SIZE * SIZE, cudaMemcpyHostToDevice);

  cudaEventCreate(&d_begin);
  cudaEventCreate(&d_end);

  cudaEventRecord(d_begin);
  kernel<<<SIZE * SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_A, d_B, d_C);
  cudaEventRecord(d_end);
  cudaMemcpy(C, d_C, sizeof(double) * SIZE * SIZE, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(d_end);

  cudaEventElapsedTime(&elapsedTime, d_begin, d_end);

  printf("Device Time: \t%f\n", elapsedTime);

  total = 0;
  for (int i = 0; i < SIZE * SIZE; i++) {
    total += C[i];
  }
  printf("Total: \t\t%f\n", total);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(A);
  free(B);
  free(C);

  return 0;
}