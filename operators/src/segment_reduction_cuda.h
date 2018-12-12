int unsorted_segment_sum_forward_gpu(
  at::Tensor data_cuda, at::Tensor segment_ids_cuda, const int* data_shape, at::Tensor output_cuda);

int unsorted_segment_sum_backward_gpu(
  at::Tensor grad_output_cuda, at::Tensor segment_ids_cuda, const int* data_shape, at::Tensor grad_data_cuda);

