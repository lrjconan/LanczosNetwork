#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <THC/THC.h>
// #include <torch/torch.h>
#include <torch/extension.h>

#include "cuda/segment_reduction.h"

extern THCState *state;


int unsorted_segment_sum_forward_gpu(
  at::Tensor data_cuda, at::Tensor segment_ids_cuda, const int* data_shape, at::Tensor output_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* data        = data_cuda.data<float>();
  long*  segment_ids = segment_ids_cuda.data<long>();
  float* output      = output_cuda.data<float>();

  unsorted_segment_sum_forward_gpu_kernel_launcher(
    stream, data, segment_ids, data_shape, output);

  return 1;
}

int unsorted_segment_sum_backward_gpu(
  at::Tensor grad_output_cuda, at::Tensor segment_ids_cuda, const int* data_shape, at::Tensor grad_data_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* grad_output = grad_output_cuda.data<float>();
  long*  segment_ids = segment_ids_cuda.data<long>();
  float* grad_data   = grad_data_cuda.data<float>();

  unsorted_segment_sum_backward_gpu_kernel_launcher(
    stream, grad_output, segment_ids, data_shape, grad_data);

  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &unsorted_segment_sum_forward_gpu, "UnsortedSegSum forward (CUDA)");
    m.def("backward", &unsorted_segment_sum_backward_gpu, "UnsortedSegSum backward (CUDA)");
}
