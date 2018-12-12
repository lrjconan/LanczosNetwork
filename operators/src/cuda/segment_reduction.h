#ifndef SEGMENT_REDUCTION_KERNEL
#define SEGMENT_REDUCTION_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void unsorted_segment_sum_forward_gpu_kernel_launcher(
  cudaStream_t stream, const float* data, const long* segment_ids, const int* data_shape, float* output);

void unsorted_segment_sum_backward_gpu_kernel_launcher(
  cudaStream_t stream, const float* grad_output, const long* segment_ids, const int* data_shape, float* grad_data);


#ifdef __cplusplus
}
#endif

#endif
