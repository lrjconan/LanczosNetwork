import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from operators.functions.unsorted_segment_sum import UnsortedSegmentSumFunction


class UnsortedSegmentSum(nn.Module):

  def __init__(self, num_segments):
    super(UnsortedSegmentSum, self).__init__()
    self.num_segments = num_segments

  def forward(self, data, segment_index):
    return UnsortedSegmentSumFunction.apply(data, segment_index,
                                            self.num_segments)
