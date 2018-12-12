import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.set2set import Set2Vec
from operators.functions.unsorted_segment_sum import UnsortedSegmentSumFunction

EPS = float(np.finfo(np.float32).eps)
unsorted_segment_sum = UnsortedSegmentSumFunction.apply

__all__ = ['MPNN']


class MPNN(nn.Module):

  def __init__(self, config):
    """ Message Passing Neural Networks,
        see reference below for more information

        Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O. and Dahl,
        G.E., 2017. Neural message passing for quantum chemistry. In ICML.
    """
    super(MPNN, self).__init__()
    self.config = config
    self.input_dim = config.model.input_dim
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = config.model.output_dim
    self.num_layer = config.model.num_layer
    self.num_prop = config.model.num_prop
    self.msg_func_name = config.model.msg_func
    self.num_step_set2vec = config.model.num_step_set2vec
    self.dropout = config.model.dropout if hasattr(config.model,
                                                   'dropout') else 0.0
    self.num_atom = config.dataset.num_atom
    self.num_edgetype = config.dataset.num_bond_type
    self.aggregate_type = config.model.aggregate_type
    assert self.num_layer == 1, 'not implemented'
    assert self.aggregate_type in ['avg', 'sum'], 'not implemented'

    self.node_embedding = nn.Embedding(self.num_atom, self.input_dim)

    # input function
    self.input_func = nn.Sequential(
        *[nn.Linear(self.input_dim, self.hidden_dim)])

    # update function
    self.update_func = nn.GRUCell(
        input_size=self.hidden_dim * (self.num_edgetype + 1),
        hidden_size=self.hidden_dim)

    # message function
    # N.B.: if there is no edge feature, the edge network in the paper degenerates
    # to multiple edge embedding matrices, each corresponds to one edge type
    if config.model.msg_func == 'embedding':
      self.edge_embedding = nn.Embedding(self.num_edgetype + 1, self.hidden_dim
                                         **2)
    elif config.model.msg_func == 'MLP':
      self.edge_func = nn.ModuleList([
          nn.Sequential(*[
              nn.Linear(self.hidden_dim * 2, 64),
              nn.ReLU(),
              nn.Linear(64, self.hidden_dim)
          ]) for _ in range((self.num_edgetype + 1))
      ])
    else:
      raise ValueError('Non-supported message function')

    self.att_func = Set2Vec(self.hidden_dim, self.num_step_set2vec)

    # output function
    self.output_func = nn.Sequential(
        *[nn.Linear(2 * self.hidden_dim, self.output_dim)])

    if config.model.loss == 'CrossEntropy':
      self.loss_func = torch.nn.CrossEntropyLoss()
    elif config.model.loss == 'MSE':
      self.loss_func = torch.nn.MSELoss()
    elif config.model.loss == 'L1':
      self.loss_func = torch.nn.L1Loss()
    else:
      raise ValueError("Non-supported loss function!")

    self._init_param()

  def _init_param(self):
    mlp_modules = [
        xx for xx in [self.input_func, self.output_func, self.att_func]
        if xx is not None
    ]

    for m in mlp_modules:
      if isinstance(m, nn.Sequential):
        for mm in m:
          if isinstance(mm, nn.Linear):
            nn.init.xavier_uniform_(mm.weight.data)
            if mm.bias is not None:
              mm.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

    for m in [self.update_func]:
      nn.init.xavier_uniform_(m.weight_hh.data)
      nn.init.xavier_uniform_(m.weight_ih.data)
      if m.bias:
        m.bias_hh.data.zero_()
        m.bias_ih.data.zero_()

  def forward(self, node_feat, L, label=None, mask=None):
    L[L != 0] = 1.0
    batch_size = node_feat.shape[0]
    num_node = node_feat.shape[1]
    state = self.node_embedding(node_feat)  # shape: B X N X D
    state = self.input_func(state)

    if self.msg_func_name == 'MLP':
      idx_row, idx_col = np.meshgrid(range(num_node), range(num_node))
      idx_row, idx_col = idx_row.flatten().astype(
          np.int64), idx_col.flatten().astype(np.int64)

    def _prop(state_old):
      state_dim = state_old.shape[2]

      msg = []
      for ii in range(self.num_edgetype + 1):
        if self.msg_func_name == 'embedding':
          idx_edgetype = torch.Tensor([ii]).long()
          if node_feat.is_cuda:
            idx_edgetype = idx_edgetype.cuda()
          edge_em = self.edge_embedding(idx_edgetype).view(state_dim, state_dim)
          node_state = state_old.view(batch_size * num_node,
                                      -1)  # shape: BN X D
          tmp_msg = node_state.mm(edge_em).view(batch_size, num_node,
                                                -1)  # shape: B X N X D
          # aggregate message
          if self.aggregate_type == 'sum':
            tmp_msg = torch.bmm(L[:, :, :, ii], tmp_msg)
          elif self.aggregate_type == 'avg':
            denom = torch.sum(L[:, :, :, ii], dim=2, keepdim=True) + EPS
            tmp_msg = torch.bmm(L[:, :, :, ii] / denom, tmp_msg)
          else:
            pass

        elif self.msg_func_name == 'MLP':
          state_in = state_old[:, idx_col, :]  # shape: B X N X D
          state_out = state_old[:, idx_row, :]  # shape: B X N X D
          tmp_msg = self.edge_func[ii](torch.cat(
              [state_out, state_in], dim=2).view(
                  batch_size * num_node * num_node, -1)).view(
                      batch_size, num_node, num_node,
                      -1)  # shape: B X N X N X D

          # aggregate message
          if self.aggregate_type == 'sum':
            tmp_msg = torch.matmul(
                tmp_msg.permute(0, 1, 3, 2),
                L[:, :, :, ii].unsqueeze(dim=3)).squeeze()  # B X N X D
          elif self.aggregate_type == 'avg':
            denom = torch.sum(
                L[:, :, :, ii], dim=2, keepdim=True) + EPS  # B X N X 1
            tmp_msg = torch.matmul(
                tmp_msg.permute(0, 1, 3, 2),
                L[:, :, :, ii].unsqueeze(dim=3)).squeeze()  # B X N X D
            tmp_msg = tmp_msg / denom
          else:
            pass

        msg += [tmp_msg]  # shape B X N X D

      # update state
      msg = torch.cat(msg, dim=2).view(batch_size * num_node, -1)
      state_old = state_old.view(batch_size * num_node, -1)

      # GRU update
      state_new = self.update_func(msg, state_old).view(batch_size, num_node,
                                                        -1)

      return state_new

    # propagation
    for tt in range(self.num_prop):
      state = _prop(state)
      state = F.dropout(state, self.dropout, training=self.training)

    # output
    y = []
    if mask is not None:
      for bb in range(batch_size):
        y += [self.att_func(state[bb, mask[bb], :])]
    else:
      for bb in range(batch_size):
        y += [self.att_func(state[bb, :, :])]

    score = self.output_func(torch.cat(y, dim=0))

    if label is not None:
      return score, self.loss_func(score, label)
    else:
      return score
