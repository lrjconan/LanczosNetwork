import math
import numpy as np
import torch
from torch.autograd import Function, Variable
from operators._ext import segment_reduction


class UnsortedSegmentSumFunction(Function):

  @staticmethod
  def forward(ctx, data, segment_index, num_segments):

    # data's shape should be (batch, dim1, dim2), and segment reduction will be performed over dim1

    ctx.save_for_backward(data, segment_index)
    # data = data.contiguous()
    # segment_index = segment_index.contiguous()

    if not data.is_cuda:
      output = torch.FloatTensor(data.size(0), num_segments,
                                 data.size(2)).zero_()
      segment_reduction.unsorted_segment_sum_forward(data, segment_index,
                                                     data.size(), output)
    else:
      output = torch.cuda.FloatTensor(data.size(0), num_segments,
                                      data.size(2)).zero_()
      segment_reduction.unsorted_segment_sum_forward_gpu(data, segment_index,
                                                         data.size(), output)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    data, segment_index = ctx.saved_tensors
    grad_data = data.new().resize_as_(data).zero_()

    if not data.is_cuda:
      segment_reduction.unsorted_segment_sum_backward(
          grad_output.data, segment_index, grad_data.size(), grad_data)
    else:
      segment_reduction.unsorted_segment_sum_backward_gpu(
          grad_output.data, segment_index, grad_data.size(), grad_data)

    return Variable(grad_data), None, None
