#include <cuda.h>
#include <stdio.h>

#define RUNTIME_API_CALL(apiFuncCall)                                          \
  do {                                                                         \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",     \
              __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));  \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

int main() {
  int device_count;
  int current_device;
  cudaDeviceProp devProp;

  RUNTIME_API_CALL(cudaGetDeviceCount(&device_count));
  RUNTIME_API_CALL(cudaGetDevice(&current_device));

  printf("\nNumber of devices: %d\n", device_count);
  for (int i = 0; i < device_count; i++) {
    RUNTIME_API_CALL(cudaGetDeviceProperties(&devProp, i));
    if (i == current_device) {
      printf("\nDevice: %d (Current Device)\n", i);
    } else {
      printf("\nDevice: %d\n", i);
    }
    printf("- Device Name: %s\n", devProp.name);
    printf("- Compute Capability: %d.%d\n", devProp.major, devProp.minor);
    printf("- Max Grid Size: (%d, %d, %d)\n", devProp.maxGridSize[0],
           devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf("- Max Block Size: (%d, %d, %d)\n", devProp.maxThreadsDim[0],
           devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("- Max Threads per Block: %d\n", devProp.maxThreadsPerBlock);
    printf("- Warp Size: %d\n", devProp.warpSize);
  }

  printf("\n");
  return 0;
}
