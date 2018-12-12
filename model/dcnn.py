import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DCNN']


class DCNN(nn.Module):
  """ Diffusion-convolutional neural networks,
      see reference below for more information

      Atwood, J. and Towsley, D., 2016.
      Diffusion-convolutional neural networks. In NIPS.
  """

  def __init__(self, config):
    super(DCNN, self).__init__()
    self.config = config
    self.input_dim = config.model.input_dim
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = config.model.output_dim
    self.num_layer = config.model.num_layer
    self.diffusion_dist = config.model.diffusion_dist
    self.num_scale = len(self.diffusion_dist)
    self.max_dist = max(config.model.diffusion_dist)
    self.num_atom = config.dataset.num_atom
    self.num_edgetype = config.dataset.num_bond_type
    self.dropout = config.model.dropout if hasattr(config.model,
                                                   'dropout') else 0.0

    dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
    self.filter = nn.ModuleList([
        nn.Linear(dim_list[tt] * (self.num_scale + self.num_edgetype + 1),
                  dim_list[tt + 1]) for tt in range(self.num_layer)
    ] + [nn.Linear(dim_list[-2], dim_list[-1])])

    self.embedding = nn.Embedding(self.num_atom, self.input_dim)

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

  def forward(self, node_feat, L, label=None, mask=None):
    """
      shape parameters:
      batch size = B, total number of nodes per mini-batch = N_hat
      embedding dim = D, hidden dim = H, num edge types = C
      num property = P

      node_feat: long tensor, shape B X N
      L: float tensor, shape B X N X N
      label: float tensor, shape B X P
      mask: float tensor, shape B X N
    """
    batch_size = node_feat.shape[0]
    num_node = node_feat.shape[1]
    state = self.embedding(node_feat)  # shape: B X N X D

    # propagation
    for tt in range(self.num_layer):
      state_scale = []
      tmp_state = state
      for ii in range(1, self.max_dist + 1):
        tmp_state = torch.bmm(L[:, :, :, 0], tmp_state)
        if ii in self.diffusion_dist:
          state_scale += [tmp_state]

      msg = []
      for ii in range(self.num_edgetype + 1):
        msg += [torch.bmm(L[:, :, :, ii], state)]  # shape: B X N X D

      msg = torch.cat(msg + state_scale, dim=2).view(num_node * batch_size, -1)
      state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
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
