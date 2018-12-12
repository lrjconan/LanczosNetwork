int unsorted_segment_sum_forward(
  at::Tensor data, at::Tensor segment_ids, const int* data_shape, at::Tensor output);

int unsorted_segment_sum_backward(
  at::Tensor grad_output, at::Tensor segment_ids, const int* data_shape, at::Tensor grad_data);

