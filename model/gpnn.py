import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GPNN']
EPS = float(np.finfo(np.float32).eps)


class GPNN(nn.Module):

  def __init__(self, config):
    """ Graph Partition Neural Networks,
        see reference below for more information

        Liao, R., Brockschmidt, M., Tarlow, D., Gaunt, A.L., 
        Urtasun, R. and Zemel, R., 2018. 
        Graph Partition Neural Networks for Semi-Supervised 
        Classification. arXiv preprint arXiv:1803.06272.
    """

    super(GPNN, self).__init__()
    self.config = config
    self.input_dim = config.model.input_dim
    self.hidden_dim = config.model.hidden_dim
    self.output_dim = config.model.output_dim
    self.num_layer = config.model.num_layer
    self.num_prop = config.model.num_prop
    self.num_partition = config.model.num_partition
    self.num_prop_cluster = config.model.num_prop_cluster
    self.num_prop_cut = config.model.num_prop_cut
    self.dropout = config.model.dropout if hasattr(config.model,
                                                   'dropout') else 0.0
    self.num_atom = config.dataset.num_atom
    self.num_edgetype = config.dataset.num_bond_type
    self.aggregate_type = config.model.aggregate_type
    assert self.num_layer == 1, "not implemented"
    assert self.aggregate_type in ['avg', 'sum'], 'not implemented'

    self.embedding = nn.Embedding(self.num_atom, self.input_dim)

    # update function
    if config.model.update_func == 'RNN':
      self.update_func = nn.RNNCell(
          input_size=self.hidden_dim * (self.num_edgetype + 1),
          hidden_size=self.hidden_dim,
          nonlinearity='relu')
      self.update_func_partition = nn.RNNCell(
          input_size=self.hidden_dim,
          hidden_size=self.hidden_dim,
          nonlinearity='relu')
    elif config.model.update_func == 'GRU':
      self.update_func = nn.GRUCell(
          input_size=self.hidden_dim * (self.num_edgetype + 1),
          hidden_size=self.hidden_dim)
      self.update_func_partition = nn.GRUCell(
          input_size=self.hidden_dim, hidden_size=self.hidden_dim)
    elif config.model.update_func == 'MLP':
      self.update_func = nn.Sequential(*[
          nn.Linear(self.hidden_dim * (self.num_edgetype + 1), self.hidden_dim),
          nn.Tanh()
      ])
      self.update_func_partition = nn.Sequential(
          *[nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()])

    # state function
    self.state_func = nn.Sequential(*[
        nn.Linear(3 * self.hidden_dim, 512),
        nn.ReLU(),
        nn.Linear(512, self.hidden_dim)
    ])

    # message function
    if config.model.msg_func == 'MLP':
      self.msg_func = nn.ModuleList([
          nn.Sequential(*[
              nn.Linear(self.hidden_dim, 128),
              nn.ReLU(),
              nn.Linear(128, self.hidden_dim)
          ]) for _ in range((self.num_edgetype + 1))
      ])
    else:
      self.msg_func = None

    # attention
    self.att_func = nn.Sequential(
        *[nn.Linear(self.hidden_dim, 1),
          nn.Sigmoid()])

    self.input_func = nn.Sequential(
        *[nn.Linear(self.input_dim, self.hidden_dim)])
    self.output_func = nn.Sequential(
        *[nn.Linear(self.hidden_dim, self.output_dim)])

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
        xx for xx in [
            self.input_func, self.state_func, self.msg_func, self.att_func,
            self.output_func
        ] if xx is not None
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

    if self.config.model.update_func in ['GRU', 'RNN']:
      for m in [self.update_func, self.update_func_partition]:
        nn.init.xavier_uniform_(m.weight_hh.data)
        nn.init.xavier_uniform_(m.weight_ih.data)
        if m.bias:
          m.bias_hh.data.zero_()
          m.bias_ih.data.zero_()
    elif self.config.model.update_func == 'MLP':
      for m in [self.update_func, self.update_func_partition]:
        if isinstance(m, nn.Linear):
          nn.init.xavier_uniform_(m.weight.data)
          if m.bias is not None:
            m.bias.data.zero_()

  def forward(self, node_feat, L, L_cluster, L_cut, label=None, mask=None):
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
        L_cluster: float tensor, shape B X N X N
        L_cut: float tensor, shape B X N X N
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
    L[L != 0] = 1.0
    batch_size = node_feat.shape[0]
    num_node = node_feat.shape[1]
    state = self.embedding(node_feat)  # shape: B X N X D
    state = self.input_func(state)

    def _prop(state_old):
      # gather message
      msg = []
      for ii in range(self.num_edgetype + 1):
        if self.msg_func is not None:
          tmp_msg = self.msg_func[ii](
              state_old.view(batch_size * num_node, -1)).view(
                  batch_size, num_node, -1)  # shape: B X N X D

        # aggregate message
        if self.aggregate_type == 'sum':
          tmp_msg = torch.bmm(L[:, :, :, ii], tmp_msg)
        elif self.aggregate_type == 'avg':
          denom = torch.sum(L[:, :, :, ii], dim=2, keepdim=True) + EPS
          tmp_msg = torch.bmm(L[:, :, :, ii] / denom, tmp_msg)
        else:
          pass

        msg += [tmp_msg]  # shape B X N X D

      # update state
      msg = torch.cat(msg, dim=2).view(batch_size * num_node, -1)
      state_old = state_old.view(batch_size * num_node, -1)
      state_new = self.update_func(msg, state_old).view(batch_size, num_node,
                                                        -1)

      return state_new

    def _prop_partition(state_old, L_step):
      if self.msg_func is not None:
        # 0-th edge type corresponds to simple graph
        msg = self.msg_func[0](state_old.view(batch_size * num_node, -1)).view(
            batch_size, num_node, -1)  # shape: B X N X D

      # aggregate message
      if self.aggregate_type == 'sum':
        msg = torch.bmm(L_step, msg)
      elif self.aggregate_type == 'avg':
        denom = torch.sum(L_step, dim=2, keepdim=True) + EPS
        msg = torch.bmm(L_step / denom, msg)
      else:
        pass

      # update state
      msg = msg.view(batch_size * num_node, -1)
      state_old = state_old.view(batch_size * num_node, -1)
      state_new = self.update_func_partition(msg, state_old).view(
          batch_size, num_node, -1)

      return state_new

    # propagation
    for tt in range(self.num_prop):
      # propagate within clusters
      state_cluster = state
      for ii in range(self.num_prop_cluster):
        state_cluster = _prop_partition(state_cluster, L_cluster)

      # propagate between clusters
      state_cut = state
      for ii in range(self.num_prop_cut):
        state_cut = _prop_partition(state_cut, L_cut)

      state = self.state_func(
          torch.cat([state, state_cluster, state_cut], dim=2))
      state = _prop(state)
      state = F.dropout(state, self.dropout, training=self.training)

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
