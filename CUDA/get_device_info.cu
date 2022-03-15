#include <cuda.h>
#include <stdio.h>

int main() {
  int device_count;
  int current_device;
  int max_threads_per_block;
  cudaDeviceProp devProp;

  cudaGetDeviceCount(&device_count);

  printf("\n======== Devices Info ========\n");
  printf("Device count: %d\n", device_count);
  for (int i = 0; i < device_count; i++) {
    cudaGetDeviceProperties(&devProp, i);
    printf("Device-%d: %s\n", i, devProp.name);
  }

  cudaGetDevice(&current_device);
  cudaGetDeviceProperties(&devProp, current_device);

  max_threads_per_block = devProp.maxThreadsPerBlock;

  printf("\n======== Current Device Info ========\n");
  printf("Current device: %d\n", current_device);
  printf("Device name: %s\n", devProp.name);
  printf("Maximum threads per block: %d\n", max_threads_per_block);
  printf("\n");

  return 0;
}