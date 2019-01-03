import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GAT']


class GAT(nn.Module):
  """ Graph Attention Networks,
      see reference below for more information

      Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P.
      and Bengio, Y., 2018. Graph attention networks. In ICLR.
  """

  def __init__(self, config):
    super(GAT, self).__init__()
    self.input_dim = config.model.input_dim
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = config.model.output_dim
    self.num_layer = config.model.num_layer
    self.num_heads = config.model.num_heads
    self.dropout = config.model.dropout if hasattr(config.model,
                                                   'dropout') else 0.0
    self.num_atom = config.dataset.num_atom
    self.num_edgetype = config.dataset.num_bond_type

    self.embedding = nn.Embedding(self.num_atom, self.input_dim)

    dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
    self.filter = nn.ModuleList([
        nn.ModuleList([
            nn.ModuleList([
                nn.Linear(
                    dim_list[tt] *
                    (int(tt == 0) + int(tt != 0) * self.num_heads[tt] *
                     (self.num_edgetype + 1)),
                    dim_list[tt + 1],
                    bias=False) for _ in range(self.num_heads[tt])
            ]) for _ in range(self.num_edgetype + 1)
        ]) for tt in range(self.num_layer)
    ])

    self.att_net_1 = nn.ModuleList([
        nn.ModuleList([
            nn.ModuleList([
                nn.Linear(dim_list[tt + 1], 1)
                for _ in range(self.num_heads[tt])
            ]) for _ in range(self.num_edgetype + 1)
        ]) for tt in range(self.num_layer)
    ])

    self.att_net_2 = nn.ModuleList([
        nn.ModuleList([
            nn.ModuleList([
                nn.Linear(dim_list[tt + 1], 1)
                for _ in range(self.num_heads[tt])
            ]) for _ in range(self.num_edgetype + 1)
        ]) for tt in range(self.num_layer)
    ])

    self.state_bias = [[[None] * self.num_heads[tt]] * (self.num_edgetype + 1)
                       for tt in range(self.num_layer)]
    for tt in range(self.num_layer):
      for jj in range(self.num_edgetype + 1):
        for ii in range(self.num_heads[tt]):
          self.state_bias[tt][jj][ii] = torch.nn.Parameter(
              torch.zeros(dim_list[tt + 1]))
          self.register_parameter('bias_{}_{}_{}'.format(ii, jj, tt),
                                  self.state_bias[tt][jj][ii])

    # attention
    self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])

    self.output_func = nn.Sequential(*[nn.Linear(dim_list[-2], dim_list[-1])])

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
    for ff in self.att_func:
      if isinstance(ff, nn.Linear):
        nn.init.xavier_uniform_(ff.weight.data)
        if ff.bias is not None:
          ff.bias.data.zero_()

    for ff in self.output_func:
      if isinstance(ff, nn.Linear):
        nn.init.xavier_uniform_(ff.weight.data)
        if ff.bias is not None:
          ff.bias.data.zero_()

    for f in self.filter:
      for ff in f:
        for fff in ff:
          if isinstance(fff, nn.Linear):
            nn.init.xavier_uniform_(fff.weight.data)
            if fff.bias is not None:
              fff.bias.data.zero_()

    for f in self.att_net_1:
      for ff in f:
        for fff in ff:
          if isinstance(fff, nn.Linear):
            nn.init.xavier_uniform_(fff.weight.data)
            if fff.bias is not None:
              fff.bias.data.zero_()

    for f in self.att_net_2:
      for ff in f:
        for fff in ff:
          if isinstance(fff, nn.Linear):
            nn.init.xavier_uniform_(fff.weight.data)
            if fff.bias is not None:
              fff.bias.data.zero_()

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
    batch_size = node_feat.shape[0]
    num_node = node_feat.shape[1]
    state = self.embedding(node_feat)  # shape: B X N X D

    # propagation
    for tt in range(self.num_layer):
      h = []
      for jj in range(self.num_edgetype + 1):
        for ii in range(self.num_heads[tt]):
          state_head = F.dropout(state, self.dropout, training=self.training)
          Wh = self.filter[tt][jj][ii](
              state_head.view(batch_size * num_node, -1)).view(
                  batch_size, num_node, -1)  # B X N X D
          att_weights_1 = self.att_net_1[tt][jj][ii](Wh)  # B X N X 1
          att_weights_2 = self.att_net_2[tt][jj][ii](Wh)  # B X N X 1
          att_weights = att_weights_1 + att_weights_2.transpose(
              1, 2)  # B X N X N dense matrix

          att_weights = F.softmax(
              F.leaky_relu(att_weights, negative_slope=0.2) + L[:, :, :, jj],
              dim=1)
          att_weights = F.dropout(
              att_weights, self.dropout, training=self.training)  # B X N X N
          Wh = F.dropout(Wh, self.dropout, training=self.training)  # B X N X D

          if tt == self.num_layer - 1:
            h += [
                torch.bmm(att_weights, Wh) + self.state_bias[tt][jj][ii].view(
                    1, 1, -1)
            ]  # B X N X D
          else:
            h += [
                F.elu(
                    torch.bmm(att_weights, Wh) +
                    self.state_bias[tt][jj][ii].view(1, 1, -1))
            ]  # B X N X D

      if tt == self.num_layer - 1:
        state = torch.mean(torch.stack(h, dim=0), dim=0)  # B X N X D
      else:
        state = torch.cat(h, dim=2)

    # output
    state = state.view(batch_size * num_node, -1)
    y = self.output_func(state)  # shape: BN X 1
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
