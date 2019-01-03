import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_helper import check_dist


EPS = float(np.finfo(np.float32).eps)
__all__ = ['LanczosNet']


class LanczosNet(nn.Module):

  def __init__(self, config):
    super(LanczosNet, self).__init__()
    self.config = config
    self.input_dim = config.model.input_dim
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = config.model.output_dim
    self.num_layer = config.model.num_layer
    self.num_atom = config.dataset.num_atom
    self.num_edgetype = config.dataset.num_bond_type
    self.dropout = config.model.dropout if hasattr(config.model,
                                                   'dropout') else 0.0
    self.short_diffusion_dist = check_dist(config.model.short_diffusion_dist)
    self.long_diffusion_dist = check_dist(config.model.long_diffusion_dist)
    self.max_short_diffusion_dist = max(
        self.short_diffusion_dist) if self.short_diffusion_dist else None
    self.max_long_diffusion_dist = max(
        self.long_diffusion_dist) if self.long_diffusion_dist else None
    self.num_scale_short = len(self.short_diffusion_dist)
    self.num_scale_long = len(self.long_diffusion_dist)
    self.num_eig_vec = config.model.num_eig_vec
    self.spectral_filter_kind = config.model.spectral_filter_kind

    dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
    self.filter = nn.ModuleList([
        nn.Linear(dim_list[tt] * (
            self.num_scale_short + self.num_scale_long + self.num_edgetype + 1),
                  dim_list[tt + 1]) for tt in range(self.num_layer)
    ] + [nn.Linear(dim_list[-2], dim_list[-1])])

    self.embedding = nn.Embedding(self.num_atom, self.input_dim)

    # spectral filters
    if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
      self.spectral_filter = nn.ModuleList([
          nn.Sequential(*[
              nn.Linear(self.num_scale_long, 128),
              nn.ReLU(),
              nn.Linear(128, 128),
              nn.ReLU(),
              nn.Linear(128, 128),
              nn.ReLU(),
              nn.Linear(128, self.num_scale_long)
          ]) for _ in range(self.num_layer)
      ])

    # attention
    self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])

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
    for ff in self.filter:
      if isinstance(ff, nn.Linear):
        nn.init.xavier_uniform_(ff.weight.data)
        if ff.bias is not None:
          ff.bias.data.zero_()

    for ff in self.att_func:
      if isinstance(ff, nn.Linear):
        nn.init.xavier_uniform_(ff.weight.data)
        if ff.bias is not None:
          ff.bias.data.zero_()

    if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
      for ff in self.spectral_filter:
        for f in ff:
          if isinstance(f, nn.Linear):
            nn.init.xavier_uniform_(f.weight.data)
            if f.bias is not None:
              f.bias.data.zero_()

  def _get_spectral_filters(self, T_list, Q, layer_idx):
    """ Construct Spectral Filters based on Lanczos Outputs

      Args:
        T_list: each element is of shape B X K
        Q: shape B X N X K

      Returns:
        L: shape B X N X N X num_scale
    """
    # multi-scale diffusion
    L = []

    # spectral filter
    if self.spectral_filter_kind == 'MLP':
      DD = torch.stack(
          T_list, dim=2).view(Q.shape[0] * Q.shape[2], -1)  # shape BK X D
      DD = self.spectral_filter[layer_idx](DD).view(Q.shape[0], Q.shape[2],
                                                    -1)  # shape B X K X D
      for ii in range(self.num_scale_long):
        tmp_DD = DD[:, :, ii].unsqueeze(1).repeat(1, Q.shape[1],
                                                  1)  # shape B X N X K
        L += [(Q * tmp_DD).bmm(Q.transpose(1, 2))]
    else:
      for ii in range(self.num_scale_long):
        DD = T_list[ii].unsqueeze(1).repeat(1, Q.shape[1], 1)  # shape B X N X K
        L += [(Q * DD).bmm(Q.transpose(1, 2))]

    return torch.stack(L, dim=3)

  def forward(self, node_feat, L, D, V, label=None, mask=None):
    """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
        number of approximated eigenvalues, i.e., Ritz values = K
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        D: float tensor, Ritz values, shape B X K
        V: float tensor, Ritz vectors, shape B X N X K
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
    batch_size = node_feat.shape[0]
    num_node = node_feat.shape[1]

    D_pow_list = []

    for ii in self.long_diffusion_dist:
      D_pow_list += [torch.pow(D, ii)]  # shape B X K

    ###########################################################################
    # Graph Convolution
    ###########################################################################
    state = self.embedding(node_feat)  # shape: B X N X D
    
    # propagation
    for tt in range(self.num_layer):
      msg = []

      if self.num_scale_long > 0:
        Lf = self._get_spectral_filters(D_pow_list, V, tt)

      # short diffusion
      if self.num_scale_short > 0:
        tmp_state = state
        for ii in range(1, self.max_short_diffusion_dist + 1):
          tmp_state = torch.bmm(L[:, :, :, 0], tmp_state)
          if ii in self.short_diffusion_dist:
            msg += [tmp_state]

      # long diffusion
      if self.num_scale_long > 0:
        for ii in range(self.num_scale_long):
          msg += [torch.bmm(Lf[:, :, :, ii], state)]  # shape: B X N X D

      # edge type
      for ii in range(self.num_edgetype + 1):
        msg += [torch.bmm(L[:, :, :, ii], state)]  # shape: B X N X D

      msg = torch.cat(msg, dim=2).view(num_node * batch_size, -1)
      state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
      state = F.dropout(state, self.dropout, training=self.training)

    # output
    state = state.view(batch_size * num_node, -1)
    y = self.filter[-1](state)  # shape: BN X 1
    att_weight = self.att_func(state)  # shape: BN X 1
    y = (att_weight * y).view(batch_size, num_node, -1)

    score = []
    for bb in range(batch_size):
      score += [torch.mean(y[bb, mask[bb], :], dim=0)]

    score = torch.stack(score)

    if label is not None:
      return score, self.loss_func(score, label)
    else:
      return score
