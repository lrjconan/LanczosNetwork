import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.gnn import GNN
from memory_profiler import profile

__all__ = ['GPNN']


class GPNN(GNN):

  def __init__(self, config):
    """ Graph Partition Neural Networks,
        see reference below for more information

        Liao, R., Brockschmidt, M., Tarlow, D., Gaunt, A.L., 
        Urtasun, R. and Zemel, R., 2018. 
        Graph Partition Neural Networks for Semi-Supervised 
        Classification. arXiv preprint arXiv:1803.06272.
    """

    super(GPNN, self).__init__(config)
    self.num_partition = config.num_partition

  @profile(stream=open('./memory_profile.log', 'a'), precision=4)
  def forward(self,
              feat,
              adj_idx,
              adj_val,
              adj_shape,
              node_idx,
              label=None,
              mask=None):
    feat = feat.squeeze(dim=0)
    label = label.squeeze(dim=0) if label is not None else label
    mask = mask.squeeze(dim=0) if mask is not None else mask
    adj_idx = [ii.squeeze(dim=0) for ii in adj_idx]
    adj_val = [ii.squeeze(dim=0) for ii in adj_val]
    adj_shape = [ii.squeeze(dim=0) for ii in adj_shape]
    node_idx = [ii.squeeze(dim=0) for ii in node_idx]

    # import pdb; pdb.set_trace()
    idx_row, idx_col = [None] * (self.num_partition + 1), [None] * (
        self.num_partition + 1)
    for kk in range(self.num_partition + 1):
      idx_row[kk], idx_col[kk], node_idx[kk] = Variable(adj_idx[kk][0, :]), Variable(
          adj_idx[kk][1, :]), Variable(node_idx[kk])

    # state shape
    if self.input_func:
      state = self.input_func(feat)
    else:
      state = feat

    # propagate
    for tt in range(self.num_prop):
      # propagate on clusters & cut
      # note: for parallel implementation, put scatter_ out of for loop
      for kk in range(self.num_partition + 1):        
        state_kk = torch.index_select(state, 0, node_idx[kk])
        new_state = self._propagate(state_kk, idx_row[kk], idx_col[kk])
        state.scatter_(0, node_idx[kk].view(-1, 1).repeat(1, state.shape[1]), new_state)

    # output
    if self.output_func:
      y = self.output_func(torch.cat([state, feat], dim=1))
    else:
      y = state

    if label is not None:
      if mask is None:
        return y, self.loss_func(y, label)
      else:
        return y, self.loss_func(y[mask, :], label[mask])
    else:
      return y
