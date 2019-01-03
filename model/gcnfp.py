import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GCNFP']


class GCNFP(nn.Module):
  """ Graph Convolutional Networks for fingerprints,
      see reference below for more information

      Duvenaud, D.K., Maclaurin, D., Iparraguirre, J., Bombarell,
      R., Hirzel, T., Aspuru-Guzik, A. and Adams, R.P., 2015.
      Convolutional networks on graphs for learning molecular
      fingerprints. In NIPS.

      N.B.: the difference with GCN is, Duvenaud et. al. use
      binary adjacency matrix rather than graph Laplacian
  """

  def __init__(self, config):
    super(GCNFP, self).__init__()
    self.config = config
    self.input_dim = config.model.input_dim
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = config.model.output_dim
    self.num_layer = config.model.num_layer
    self.num_atom = config.dataset.num_atom
    self.num_edgetype = config.dataset.num_bond_type
    self.dropout = config.model.dropout if hasattr(config.model,
                                                   'dropout') else 0.0

    dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
    self.filter = nn.ModuleList([
        nn.Linear(dim_list[tt] * (self.num_edgetype + 1), dim_list[tt + 1])
        for tt in range(self.num_layer)
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
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
    L[L != 0] = 1.0
    batch_size = node_feat.shape[0]
    num_node = node_feat.shape[1]
    state = self.embedding(node_feat)  # shape: B X N X D

    # propagation
    for tt in range(self.num_layer):
      msg = []
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
