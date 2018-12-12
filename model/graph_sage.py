import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = float(np.finfo(np.float32).eps)
__all__ = ['GraphSAGE']


class GraphSAGE(nn.Module):

  def __init__(self, config):
    """ GraphSAGE,
        see reference below for more information

        Hamilton, W., Ying, Z. and Leskovec, J., 2017. Inductive
        representation learning on large graphs. In NIPS.
    """
    super(GraphSAGE, self).__init__()
    self.config = config
    self.input_dim = config.model.input_dim
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = config.model.output_dim
    self.num_layer = config.model.num_layer
    self.dropout = config.model.dropout if hasattr(config.model,
                                                   'dropout') else 0.0
    self.num_sample_neighbors = config.model.num_sample_neighbors
    self.num_atom = config.dataset.num_atom
    self.num_edgetype = config.dataset.num_bond_type
    assert self.num_layer == len(self.hidden_dim)
    dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]

    self.embedding = nn.Embedding(self.num_atom, self.input_dim)

    # aggregate function
    self.agg_func_name = config.model.agg_func
    if self.agg_func_name == 'LSTM':
      self.agg_func = nn.ModuleList([
          nn.LSTMCell(input_size=dim_list[tt], hidden_size=dim_list[tt])
          for tt in range(self.num_layer - 1)
      ])
    elif self.agg_func_name == 'Mean':
      self.agg_func = torch.mean
    elif self.agg_func_name == 'Max':
      self.agg_func = torch.max
    else:
      self.agg_func = None

    # attention
    self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])

    # update function
    self.filter = nn.ModuleList([
        nn.Linear(dim_list[tt] * (self.num_edgetype + 1), dim_list[tt + 1])
        for tt in range(self.num_layer)
    ] + [nn.Linear(dim_list[-2], dim_list[-1])])

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
    mlp_modules = [xx for xx in [self.att_func] if xx is not None]

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

    if self.config.model.agg_func == 'LSTM':
      for m in self.agg_func:
        nn.init.xavier_uniform_(m.weight_hh.data)
        nn.init.xavier_uniform_(m.weight_ih.data)
        if m.bias:
          m.bias_hh.data.zero_()
          m.bias_ih.data.zero_()

    for ff in self.filter:
      if isinstance(ff, nn.Linear):
        nn.init.xavier_uniform_(ff.weight.data)
        if ff.bias is not None:
          ff.bias.data.zero_()

  def forward(self, node_feat, nn_idx, nonempty_mask, label=None, mask=None):
    """
      node_feat: float tensor, shape B X N X D
      nn_idx: float tensor, shape B X N X K X E
      nonempty_mask: float tensor, shape B X N X 1
      label: float tensor, shape B X P
      mask: float tensor, shape B X N
    """
    batch_size = node_feat.shape[0]
    num_node = node_feat.shape[1]
    state = self.embedding(node_feat)  # shape B X N X D

    # propagation
    for ii in range(self.num_layer - 1):
      msg = []
      for jj in range(self.num_edgetype + 1):
        # gather message
        nn_state = []
        for bb in range(batch_size):
          nn_state += [state[bb, nn_idx[bb, :, :, jj], :]]  # shape N X K X D

        nn_state = torch.stack(nn_state, dim=0)  # shape B X N X K X D

        # aggregate message
        if self.agg_func_name == 'LSTM':
          cx = torch.zeros_like(state).view(batch_size * num_node,
                                            -1)  # shape: B X N X D
          hx = torch.zeros_like(state).view(batch_size * num_node, -1)
          for tt in range(self.num_sample_neighbors):
            ix = nn_state[:, :, tt, :]
            hx, cx = self.agg_func[ii](ix.view(batch_size * num_node, -1),
                                       (hx, cx))

          agg_state = hx.view(batch_size, num_node, -1)
        elif self.agg_func_name == 'Max':
          agg_state, _ = self.agg_func(nn_state, dim=2)
        else:
          agg_state = self.agg_func(nn_state, dim=2)

        msg += [agg_state * nonempty_mask]

      # update state
      # import pdb; pdb.set_trace()
      state = F.relu(self.filter[ii](torch.cat(msg, dim=2).view(
          batch_size * num_node, -1)))
      state = (state / (torch.norm(state, 2, dim=1, keepdim=True) + EPS)).view(
          batch_size, num_node, -1)
      state = F.dropout(state, self.dropout, training=self.training)

    # output
    state = state.view(batch_size * num_node, -1)
    y = self.filter[-1](state)  # shape: BN X 1
    att_weight = self.att_func(state)  # shape: BN X 1
    y = (att_weight * y).view(batch_size, num_node, -1)

    score = []
    if mask is not None:
      for bb in range(batch_size):
        score += [torch.mean(y[bb, mask[bb], :], dim=0)]
    else:
      for bb in range(batch_size):
        score += [torch.mean(y[bb, :, :], dim=0)]

    score = torch.stack(score)

    if label is not None:
      return score, self.loss_func(score, label)
    else:
      return score
