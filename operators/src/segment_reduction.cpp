#include <THC/THC.h>
// #include <torch/torch.h>
#include <torch/extension.h>


int unsorted_segment_sum_forward(
  at::Tensor data, at::Tensor segment_ids, const int* data_shape, at::Tensor output) {
  float* data_ptr        = data.data<float>();
  long*  segment_ids_ptr = segment_ids.data<long>();
  float* output_ptr      = output.data<float>();

  int dim_0 = data_shape[0];
  int dim_1 = data_shape[1];
  int dim_2 = data_shape[2];

  for(int ii = 0; ii < dim_0; ++ii)
  {
    for(int jj = 0; jj < dim_1; ++jj)
    {
      int output_idx = segment_ids_ptr[jj];

      for(int kk = 0; kk < dim_2; ++kk)
      {
        output_ptr[ii * dim_1 * dim_2 + output_idx * dim_2 + kk] += data_ptr[ii * dim_1 * dim_2 + jj * dim_2 + kk];
      }
    }
  }

  return 1;
}

int unsorted_segment_sum_backward(
  at::Tensor grad_output, at::Tensor segment_ids, const int* data_shape, at::Tensor grad_data) {

  float* grad_output_ptr = grad_output.data<float>();
  long*  segment_ids_ptr = segment_ids.data<long>();
  float* grad_data_ptr   = grad_data.data<float>();

  int dim_0 = data_shape[0];
  int dim_1 = data_shape[1];
  int dim_2 = data_shape[2];

  for(int ii = 0; ii < dim_0; ++ii)
  {
    for(int jj = 0; jj < dim_1; ++jj)
    {
      int output_idx = segment_ids_ptr[jj];

      for(int kk = 0; kk < dim_2; ++kk)
      {
        grad_data_ptr[ii * dim_1 * dim_2 + jj * dim_2 + kk] = grad_output_ptr[ii * dim_1 * dim_2 + output_idx * dim_2 + kk];
      }
    }
  }

  return 1;
}
