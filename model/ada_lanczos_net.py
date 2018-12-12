import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_helper import check_dist

EPS = float(np.finfo(np.float32).eps)
__all__ = ['AdaLanczosNet']


class AdaLanczosNet(nn.Module):

  def __init__(self, config):
    super(AdaLanczosNet, self).__init__()
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
    self.use_reorthogonalization = config.model.use_reorthogonalization if hasattr(
        config, 'use_reorthogonalization') else True
    self.use_power_iteration_cap = config.model.use_power_iteration_cap if hasattr(
        config, 'use_power_iteration_cap') else True

    dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
    self.filter = nn.ModuleList([
        nn.Linear(dim_list[tt] * (
            self.num_scale_short + self.num_scale_long + self.num_edgetype + 1),
                  dim_list[tt + 1]) for tt in range(self.num_layer)
    ] + [nn.Linear(dim_list[-2], dim_list[-1])])

    # self.embedding = nn.Embedding(self.num_atom, self.input_dim)

    self.embedding = nn.Embedding(self.num_atom, self.num_atom)
    self.embedding.weight.requires_grad = False
    self.embedding.weight.data = torch.eye(self.num_atom)
    if self.config.use_gpu:
      self.embedding.weight.data = self.embedding.weight.data.cuda()

    self.embedding_map = nn.Sequential(*[
        nn.Linear(self.num_atom, 1024),
        nn.ReLU(),
        nn.Linear(1024, self.num_atom)
    ])

    # spectral filters
    if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
      self.eigen_map = nn.ModuleList([
          nn.Sequential(*[
              nn.Linear(self.num_eig_vec * self.num_eig_vec, 1024),
              nn.ReLU(),
              nn.Linear(1024, self.num_eig_vec * self.num_eig_vec)
          ]) for _ in range(self.num_scale_long)
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

    for ff in self.eigen_map:
      if isinstance(ff, nn.Linear):
        nn.init.xavier_uniform_(ff.weight.data)
        if ff.bias is not None:
          ff.bias.data.zero_()

    if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
      for f in self.eigen_map:
        for ff in f:
          if isinstance(ff, nn.Linear):
            nn.init.xavier_uniform_(ff.weight.data)
            if ff.bias is not None:
              ff.bias.data.zero_()

  def _get_graph_laplacian(self, node_feat, adj_mask):
    """ Compute graph Laplacian

      Args:
        node_feat: float tensor, shape B X N X D
        adj_mask: float tensor, shape B X N X N, binary mask, should contain self-loop

      Returns:
        L: float tensor, shape B X N X N
    """
    batch_size = node_feat.shape[0]
    num_node = node_feat.shape[1]
    dim_feat = node_feat.shape[2]

    # compute pairwise distance
    idx_row, idx_col = np.meshgrid(range(num_node), range(num_node))
    idx_row, idx_col = torch.Tensor(idx_row.reshape(-1)).long(), torch.Tensor(
        idx_col.reshape(-1)).long()

    if node_feat.is_cuda:
      idx_row, idx_col = idx_row.cuda(), idx_col.cuda()

    diff = node_feat[:, idx_row, :] - node_feat[:,
                                                idx_col, :]  # shape B X N^2 X D
    dist2 = (diff * diff).sum(dim=2)  # shape B X N^2
    # sigma2, _ = torch.median(dist2, dim=1, keepdim=True) # median is sometimes 0

    sigma2 = torch.mean(dist2, dim=1, keepdim=True)
    A = torch.exp(-dist2 / sigma2)  # shape B X N^2
    A = A.reshape(batch_size, num_node, num_node) * adj_mask  # shape B X N X N
    row_sum = torch.sum(A, dim=2, keepdim=True)
    pad_row_sum = torch.zeros_like(row_sum)
    pad_row_sum[row_sum == 0.0] = 1.0
    # alpha = 0.25
    alpha = 0.5
    D = 1.0 / (row_sum + pad_row_sum).pow(alpha)  # shape B X N X 1
    L = D * A * D.transpose(1, 2)  # shape B X N X N

    # re-normalize for diffusion map
    # row_sum = torch.sum(L, dim=2, keepdim=True)
    # pad_row_sum = torch.zeros_like(row_sum)
    # pad_row_sum[row_sum == 0.0] = 1.0
    # D = 1.0 / (row_sum + pad_row_sum).sqrt()  # shape B X N X 1
    # L = D * L * D.transpose(1, 2)  # shape B X N X N

    return L

  def _lanczos_layer(self, A, mask=None):
    """ Lanczos layer for symmetric matrix A

    N.B.: currently we note some issues with the post-SVD when A has
      multiple similar eigenvalues

      For the mini-batch version of Lanczos, we need to intialize
      Lanczos vectors based on the individual rank of each A

      Args:
        A: float tensor, shape B X N X N
        mask: float tensor, shape B X N

      Returns:
        T: float tensor, shape B X K X K
        Q: float tensor, shape B X N X K
    """
    batch_size = A.shape[0]
    num_node = A.shape[1]
    lanczos_iter = min(num_node, self.num_eig_vec)

    # initialization
    alpha = [None] * (lanczos_iter + 1)
    beta = [None] * (lanczos_iter + 1)
    Q = [None] * (lanczos_iter + 2)

    beta[0] = torch.zeros(batch_size, 1, 1)
    Q[0] = torch.zeros(batch_size, num_node, 1)
    Q[1] = torch.randn(batch_size, num_node, 1)

    if A.is_cuda:
      Q[0], Q[1], beta[0] = Q[0].cuda(), Q[1].cuda(), beta[0].cuda()

    if mask is not None:
      mask = mask.unsqueeze(dim=2).float()
      Q[1] = Q[1] * mask

    Q[1] = Q[1] / torch.norm(Q[1], 2, dim=1, keepdim=True)

    # Lanczos loop
    lb = 1.0e-4
    valid_mask = []
    for ii in range(1, lanczos_iter + 1):
      z = torch.bmm(A, Q[ii])  # shape B X N X 1
      alpha[ii] = torch.sum(Q[ii] * z, dim=1, keepdim=True)  # shape B X 1 X 1
      z = z - alpha[ii] * Q[ii] - beta[ii - 1] * Q[ii - 1]  # shape B X N X 1

      if self.use_reorthogonalization and ii > 1:
        # N.B.: we notice gram schmidt causes instability
        def _gram_schmidt(xx, tt):
          # xx shape B X N X 1
          for jj in range(1, tt):
            xx = xx - torch.sum(
                xx * Q[jj], dim=1, keepdim=True) / (
                    torch.sum(Q[jj] * Q[jj], dim=1, keepdim=True) + EPS) * Q[jj]
          return xx

        # we do Gram Schmidt process twice
        for _ in range(2):
          z = _gram_schmidt(z, ii)

      beta[ii] = torch.norm(z, p=2, dim=1, keepdim=True)  # shape B X 1 X 1

      # N.B.: once lanczos fails at ii-th iteration, all following iterations
      # are doomed to fail
      tmp_valid_mask = (beta[ii] >= lb).float()  # shape
      if ii == 1:
        valid_mask += [tmp_valid_mask]
      else:
        valid_mask += [valid_mask[-1] * tmp_valid_mask]

      # early stop
      Q[ii + 1] = (z * valid_mask[-1]) / (beta[ii] + EPS)

    # get alpha & beta
    alpha = torch.cat(alpha[1:], dim=1).squeeze(dim=2)  # shape B X T
    beta = torch.cat(beta[1:-1], dim=1).squeeze(dim=2)  # shape B X (T-1)

    valid_mask = torch.cat(valid_mask, dim=1).squeeze(dim=2)  # shape B X T
    idx_mask = torch.sum(valid_mask, dim=1).long()
    if mask is not None:
      idx_mask = torch.min(idx_mask, torch.sum(mask, dim=1).squeeze().long())

    for ii in range(batch_size):
      if idx_mask[ii] < valid_mask.shape[1]:
        valid_mask[ii, idx_mask[ii]:] = 0.0

    # remove spurious columns
    alpha = alpha * valid_mask
    beta = beta * valid_mask[:, :-1]

    T = []
    for ii in range(batch_size):
      T += [
          torch.diag(alpha[ii]) + torch.diag(beta[ii], diagonal=1) + torch.diag(
              beta[ii], diagonal=-1)
      ]

    T = torch.stack(T, dim=0)  # shape B X T X T
    Q = torch.cat(Q[1:-1], dim=2)  # shape B X N X T
    Q_mask = valid_mask.unsqueeze(dim=1).repeat(1, Q.shape[1], 1)

    # remove spurious rows
    for ii in range(batch_size):
      if idx_mask[ii] < Q_mask.shape[1]:
        Q_mask[ii, idx_mask[ii]:, :] = 0.0

    Q = Q * Q_mask

    # pad 0 when necessary
    if lanczos_iter < self.num_eig_vec:
      pad = (0, self.num_eig_vec - lanczos_iter, 0,
             self.num_eig_vec - lanczos_iter)
      T = F.pad(T, pad)
      pad = (0, self.num_eig_vec - lanczos_iter)
      Q = F.pad(Q, pad)

    return T, Q

  def _get_spectral_filters(self, T, Q):
    """ Construct Spectral Filters based on Lanczos Outputs

      Args:
        T: shape B X K X K
        Q: shape B X N X K

      Returns:
        L: shape B X N X N X num_scale
    """
    # multi-scale diffusion
    L = []
    T_list = []
    TT = T

    b = torch.randn(T.shape[0], T.shape[1], 1)
    if T.is_cuda:
      b = b.cuda()
    Tb = b / torch.norm(b, 2, dim=1, keepdim=True)  # shape B X K X 1

    l_max_tol = 1.0
    # N.B.: while computing the matrix power, we can also run a power-iteration
    # to estimate largest eigenvalue of tri-diagonal matrix T
    for ii in range(self.max_long_diffusion_dist + 1):
      if ii in self.long_diffusion_dist:
        T_list += [TT]

      TT = torch.bmm(TT, T)  # shape B X K X K

      if self.use_power_iteration_cap:
        Tb_new = torch.bmm(T, Tb)  # shape B X K X 1
        l_max = torch.sum(
            Tb_new * Tb, dim=1, keepdim=True) / (
                torch.sum(Tb * Tb, dim=1, keepdim=True) + EPS)
        l_mask = (l_max.abs() <= l_max_tol).float()
        TT = TT * l_mask
        Tb = Tb_new / (torch.norm(Tb_new, 2, dim=1, keepdim=True) + EPS
                      )  # shape B X K X 1

    # spectral filter
    triu_mask = torch.triu(torch.ones_like(T[0]))  # shape K X K
    diag_mask = torch.diag(torch.diag(torch.ones_like(T[0])))  # shape K X K
    triu_mask = triu_mask.unsqueeze(0).repeat(T.shape[0], 1,
                                              1)  # shape B X K X K
    diag_mask = diag_mask.unsqueeze(0).repeat(T.shape[0], 1,
                                              1)  # shape B X K X K

    for ii in range(self.num_scale_long):
      if self.spectral_filter_kind == 'MLP':
        DD = self.eigen_map[ii](T_list[ii].view(T.shape[0], -1)).view(
            T.shape[0], T.shape[1], T.shape[2])  # shape: B X K X K

        # construct symmetric output
        triu_DD = DD * triu_mask
        diag_DD = DD * diag_mask
        DD = triu_DD + triu_DD.transpose(1, 2) - diag_DD

        # make DD PSD
        # DD = torch.bmm(DD, DD.transpose(1, 2))
      else:
        DD = T_list[ii]

      L += [Q.bmm(DD).bmm(Q.transpose(1, 2))]

    # if self.spectral_filter_kind == 'MLP':
    #   triu_mask = torch.triu(torch.ones_like(T[0])) # shape K X K
    #   diag_mask = torch.diag(torch.diag(torch.ones_like(T[0]))) # shape K X K
    #   triu_mask = triu_mask.unsqueeze(0).repeat(T.shape[0], 1, 1) # shape B X K X K
    #   diag_mask = diag_mask.unsqueeze(0).repeat(T.shape[0], 1, 1) # shape B X K X K
    #   TTT = torch.stack(T_list, dim=3).view(T.shape[0]*T.shape[1], -1) # shape BK X KD
    #   TTT = self.eigen_map(TTT).view(T.shape[0], T.shape[1], T.shape[2], -1) # shape B X K X K X D

    #   for ii in range(self.num_scale_long):
    #     # construct symmetric output
    #     triu_DD = TTT[:,:,:,ii] * triu_mask
    #     diag_DD = TTT[:,:,:,ii] * diag_mask
    #     DD = triu_DD + triu_DD.transpose(1,2) - diag_DD

    #     # make DD PSD
    #     # DD = torch.bmm(DD, DD.transpose(1, 2))

    #     L += [Q.bmm(DD).bmm(Q.transpose(1, 2))]
    # else:
    #   for ii in range(self.num_scale_long):
    #     L += [Q.bmm(T_list[ii]).bmm(Q.transpose(1, 2))]

    return torch.stack(L, dim=3)

  def forward(self, node_feat, L, label=None, mask=None):
    """
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
    batch_size = node_feat.shape[0]
    num_node = node_feat.shape[1]
    input_state = self.embedding(node_feat)  # shape: B X N X D
    # graph_state = self.graph_embedding(node_feat)

    input_state = self.embedding_map(input_state)
    graph_state = input_state

    if self.num_scale_long > 0:
      # compute graph Laplacian for simple graph
      adj = torch.zeros_like(L[:, :, :, 0])  # get L of simple graph
      adj[L[:, :, :, 0] != 0.0] = 1.0
      Le = self._get_graph_laplacian(graph_state, adj)

      # Lanczos Iteration
      T, Q = self._lanczos_layer(Le, mask)
      Lf = self._get_spectral_filters(T, Q)

    ###########################################################################
    # Graph Convolution
    ###########################################################################
    # propagation
    state = input_state
    for tt in range(self.num_layer):
      msg = []

      # short diffusion
      if self.num_scale_short > 0:
        tmp_state = state
        for ii in range(1, self.max_short_diffusion_dist + 1):
          tmp_state = torch.bmm(L[:, :, :, 0], tmp_state)
          # tmp_state = torch.bmm(Le, tmp_state)
          if ii in self.short_diffusion_dist:
            msg += [tmp_state]

      # long diffusion
      if self.num_scale_long > 0:
        for ii in range(self.num_scale_long):
          msg += [torch.bmm(Lf[:, :, :, ii], state)]  # shape: B X N X D

      # edge type
      for ii in range(self.num_edgetype + 1):
        msg += [torch.bmm(L[:, :, :, ii], state)]  # shape: B X N X D

      # msg += [Q]
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
