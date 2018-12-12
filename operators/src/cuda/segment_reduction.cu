#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <float.h>

#include "segment_reduction.h"

#ifdef __cplusplus
extern "C" {
#endif

const int kBaseThreadBits = 8;
const int kBaseThreadNum  = 1 << kBaseThreadBits;
const int kMaxGridNum = 65535;

int cuda_get_num_blocks(const int N) {
  return kMaxGridNum < ((N + kBaseThreadNum - 1) / kBaseThreadNum) ? kMaxGridNum : ((N + kBaseThreadNum - 1) / kBaseThreadNum);
}

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK(x) \
  /* Code block avoids redefinition of cudaError_t err */ \
  do { \
    cudaError err = cudaPeekAtLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Name: %s ErrStr:%s", #x, cudaGetErrorString(err)); \
        exit(-1); \
    } \
  } while (0);


__global__ void unsorted_segment_sum_forward_gpu_kernel(const int n, const float* data, const long* segment_ids,
  const int num_batch, const int dim1, const int dim2,
  float* output) {
  CUDA_KERNEL_LOOP(index, n) {
    int x = index % dim2;
    int c = index / dim2 % dim1;
    int b = index / dim2 / dim1;
    // data layout: b * c * x

    int reduction_pos = b * dim1 * dim2 + segment_ids[b * dim1 + c] * dim2 + x;

    atomicAdd(output + reduction_pos, data[index]);

  }
}

__global__ void unsorted_segment_sum_backward_gpu_kernel(const int n, const float* grad_output, const long* segment_ids,
  const int num_batch, const int dim1, const int dim2,
  float* grad_data) {
  CUDA_KERNEL_LOOP(index, n) {
    int x = index % dim2;
    int c = index / dim2 % dim1;
    int b = index / dim2 / dim1;
    // grad_data layout: b * c * x

    int reduction_pos = b * dim1 * dim2 + segment_ids[b * dim1 + c] * dim2 + x;

    grad_data[index] = grad_output[reduction_pos];

  }
}


void unsorted_segment_sum_forward_gpu_kernel_launcher(
  cudaStream_t stream, const float* data, const long* segment_ids, const int* data_shape, float* output) {

  uint32_t num_kernels = data_shape[0] * data_shape[1] * data_shape[2];

  unsorted_segment_sum_forward_gpu_kernel
      <<<cuda_get_num_blocks(num_kernels), kBaseThreadNum, 0, stream>>>(
      num_kernels, data, segment_ids, data_shape[0], data_shape[1], data_shape[2], output);
  CUDA_POST_KERNEL_CHECK(unsorted_segment_sum_forward_gpu_kernel);

}

void unsorted_segment_sum_backward_gpu_kernel_launcher(
  cudaStream_t stream, const float* grad_output, const long* segment_ids, const int* data_shape, float* grad_data) {


  uint32_t num_kernels = data_shape[0] * data_shape[1] * data_shape[2];

  unsorted_segment_sum_backward_gpu_kernel
      <<<cuda_get_num_blocks(num_kernels), kBaseThreadNum, 0, stream>>>(
      num_kernels, grad_output, segment_ids, data_shape[0], data_shape[1], data_shape[2], grad_data);
  CUDA_POST_KERNEL_CHECK(unsorted_segment_sum_backward_gpu_kernel);

}


#ifdef __cplusplus
}
#endif
