import os
import glob
import torch
import pickle
import numpy as np
from utils.spectral_graph_partition import *

__all__ = ['QM8Data']


class QM8Data(object):

  def __init__(self, config, split='train'):
    assert split == 'train' or split == 'dev' or split == 'test', "no such split"
    self.split = split
    self.config = config
    self.seed = config.seed
    self.npr = np.random.RandomState(self.seed)
    self.data_path = config.dataset.data_path
    self.num_edgetype = config.dataset.num_bond_type
    self.model_name = config.model.name
    self.use_eigs = True if hasattr(config.model, 'num_eig_vec') else False
    if self.use_eigs:
      self.num_eigs = config.model.num_eig_vec

    if self.model_name == 'GraphSAGE':
      self.num_sample_neighbors = config.model.num_sample_neighbors

    self.train_data_files = glob.glob(
        os.path.join(self.data_path, 'QM8_preprocess_train_*.p'))
    self.dev_data_files = glob.glob(
        os.path.join(self.data_path, 'QM8_preprocess_dev_*.p'))
    self.test_data_files = glob.glob(
        os.path.join(self.data_path, 'QM8_preprocess_test_*.p'))

    self.num_train = len(self.train_data_files)
    self.num_dev = len(self.dev_data_files)
    self.num_test = len(self.test_data_files)
    self.num_graphs = self.num_train + self.num_dev + self.num_test

  def __getitem__(self, index):
    if self.split == 'train':
      return pickle.load(open(self.train_data_files[index], 'rb'))
    elif self.split == 'dev':
      return pickle.load(open(self.dev_data_files[index], 'rb'))
    else:
      return pickle.load(open(self.test_data_files[index], 'rb'))

  def __len__(self):
    if self.split == 'train':
      return self.num_train
    elif self.split == 'dev':
      return self.num_dev
    else:
      return self.num_test

  def collate_fn(self, batch):
    """
      Collate function for mini-batch
      N.B.: we pad all samples to the maximum of the mini-batch
    """
    assert isinstance(batch, list)

    data = {}
    batch_size = len(batch)
    node_size = [bb['node_feat'].shape[0] for bb in batch]
    batch_node_size = max(node_size)  # value -> N
    pad_node_size = [batch_node_size - nn for nn in node_size]

    # pad feature: shape (B, N)
    data['node_feat'] = torch.stack([
        torch.from_numpy(
            np.pad(
                bb['node_feat'], (0, pad_node_size[ii]),
                'constant',
                constant_values=0.0)) for ii, bb in enumerate(batch)
    ]).long()

    # binary mask: shape (B, N)
    data['node_mask'] = torch.stack([
        torch.from_numpy(
            np.pad(
                np.ones(node_size[ii]), (0, pad_node_size[ii]),
                'constant',
                constant_values=0.0)) for ii, bb in enumerate(batch)
    ]).byte()

    # label: shape (B, O)
    data['label'] = torch.cat(
        [torch.from_numpy(bb['label']) for bb in batch], dim=0).float()

    if self.model_name == 'GPNN':
      #########################################################################
      # GPNN
      # N.B.: one can perform graph partition offline to speed up
      #########################################################################      
      # graph Laplacian of multi-graph: shape (B, N, N, E)
      L_multi = np.stack(
          [
              np.pad(
                  bb['L_multi'], ((0, pad_node_size[ii]),
                                  (0, pad_node_size[ii]), (0, 0)),
                  'constant',
                  constant_values=0.0) for ii, bb in enumerate(batch)
          ],
          axis=0)

      # graph Laplacian of simple graph: shape (B, N, N, 1)
      L_simple = np.stack(
          [
              np.expand_dims(
                  np.pad(
                      bb['L_simple_4'], (0, pad_node_size[ii]),
                      'constant',
                      constant_values=0.0),
                  axis=3) for ii, bb in enumerate(batch)
          ],
          axis=0)

      L = np.concatenate([L_simple, L_multi], axis=3)
      data['L'] = torch.from_numpy(L).float()

      # graph partition
      L_cluster, L_cut = [], []

      for ii in range(batch_size):
        node_label = spectral_clustering(L_simple[ii, :, :, 0], self.config.model.num_partition)
        
        # Laplacian of clusters and cut
        L_cluster_tmp, L_cut_tmp = get_L_cluster_cut(L_simple[ii, :, :, 0], node_label)

        L_cluster += [L_cluster_tmp]
        L_cut += [L_cut_tmp]

      data['L_cluster'] = torch.from_numpy(np.stack(L_cluster, axis=0)).float()
      data['L_cut'] = torch.from_numpy(np.stack(L_cut, axis=0)).float()
    elif self.model_name == 'GraphSAGE':
      #########################################################################
      # GraphSAGE
      #########################################################################
      # N.B.: adjacency mat of GraphSAGE is asymmetric
      nonempty_mask = np.zeros((batch_size, batch_node_size, 1))
      nn_idx = np.zeros((batch_size, batch_node_size, self.num_sample_neighbors,
                         self.num_edgetype + 1))

      for ii in range(batch_size):
        for jj in range(self.num_edgetype + 1):
          if jj == 0:
            tmp_L = batch[ii]['L_simple_4']
          else:
            tmp_L = batch[ii]['L_multi'][:, :, jj - 1]

          for nn in range(tmp_L.shape[0]):
            nn_list = np.nonzero(tmp_L[nn, :])[0]

            if len(nn_list) >= self.num_sample_neighbors:
              nn_idx[ii, nn, :, jj] = self.npr.choice(
                  nn_list, size=self.num_sample_neighbors, replace=False)
              nonempty_mask[ii, nn] = 1
            elif len(nn_list) > 0:
              nn_idx[ii, nn, :, jj] = self.npr.choice(
                  nn_list, size=self.num_sample_neighbors, replace=True)
              nonempty_mask[ii, nn] = 1

      data['nn_idx'] = torch.from_numpy(nn_idx).long()
      data['nonempty_mask'] = torch.from_numpy(nonempty_mask).float()
    elif self.model_name == 'GAT':
      #########################################################################
      # GAT
      #########################################################################
      # graph Laplacian of multi-graph: shape (B, N, N, E)
      L_multi = np.stack(
          [
              np.pad(
                  bb['L_multi'], ((0, pad_node_size[ii]),
                                  (0, pad_node_size[ii]), (0, 0)),
                  'constant',
                  constant_values=0.0) for ii, bb in enumerate(batch)
          ],
          axis=0)

      # graph Laplacian of simple graph: shape (B, N, N, 1)
      L_simple = np.stack(
          [
              np.expand_dims(
                  np.pad(
                      bb['L_simple_4'], (0, pad_node_size[ii]),
                      'constant',
                      constant_values=0.0),
                  axis=3) for ii, bb in enumerate(batch)
          ],
          axis=0)

      L = np.concatenate([L_simple, L_multi], axis=3)

      # trick of graph attention networks
      def adj_to_bias(adj, sizes, nhood=1):
        nb_graphs = adj.shape[0]
        mt = np.empty(adj.shape)
        for g in range(nb_graphs):
          mt[g] = np.eye(adj.shape[1])
          for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
          for i in range(sizes[g]):
            for j in range(sizes[g]):
              if mt[g][i][j] > 0.0:
                mt[g][i][j] = 1.0
        return -1e9 * (1.0 - mt)

      L_new = []
      for ii in range(batch_size):
        L_new += [
            np.transpose(
                adj_to_bias(
                    np.transpose(L[ii, :, :, :], (2, 0, 1)),
                    [batch_node_size] * L.shape[3]), (1, 2, 0))
        ]

      data['L'] = torch.from_numpy(np.stack(L_new, axis=0)).float()
    else:
      #########################################################################
      # All other models
      #########################################################################      
      # graph Laplacian of multi-graph: shape (B, N, N, E)
      L_multi = torch.stack([
          torch.from_numpy(
              np.pad(
                  bb['L_multi'], ((0, pad_node_size[ii]),
                                  (0, pad_node_size[ii]), (0, 0)),
                  'constant',
                  constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()

      # graph Laplacian of simple graph: shape (B, N, N, 1)
      L_simple_key = 'L_simple_4'
      if self.model_name == 'DCNN':
        L_simple_key = 'L_simple_7'
      elif self.model_name in ['ChebyNet']:
        L_simple_key = 'L_simple_6'

      if self.model_name == 'ChebyNet':
        L_simple = torch.stack([
            torch.from_numpy(
                np.expand_dims(
                    np.pad(
                        -bb[L_simple_key], (0, pad_node_size[ii]),
                        'constant',
                        constant_values=0.0),
                    axis=3)) for ii, bb in enumerate(batch)
        ]).float()
      else:
        L_simple = torch.stack([
            torch.from_numpy(
                np.expand_dims(
                    np.pad(
                        bb[L_simple_key], (0, pad_node_size[ii]),
                        'constant',
                        constant_values=0.0),
                    axis=3)) for ii, bb in enumerate(batch)
        ]).float()

      data['L'] = torch.cat([L_simple, L_multi], dim=3)

      # eigenvalues & eigenvectors of simple graph
      if self.use_eigs:
        eigs, eig_vecs = [], []
        for ii, bb in enumerate(batch):
          pad_eigs_len = self.num_eigs - len(bb['D_simple'])
          eigs += [
              bb['D_simple'][:self.num_eigs] if pad_eigs_len <= 0 else np.pad(
                  bb['D_simple'], (0, pad_eigs_len),
                  'constant',
                  constant_values=0.0)
          ]

          # pad eigenvectors
          pad_eig_vec = np.pad(
              bb['V_simple'], ((0, pad_node_size[ii]), (0, 0)),
              'constant',
              constant_values=0.0)

          eig_vecs += [
              pad_eig_vec[:, :self.num_eigs] if pad_eigs_len <= 0 else np.pad(
                  pad_eig_vec, ((0, 0), (0, pad_eigs_len)),
                  'constant',
                  constant_values=0.0)
          ]

        data['D'] = torch.stack([torch.from_numpy(ee) for ee in eigs]).float()
        data['V'] = torch.stack(
            [torch.from_numpy(vv) for vv in eig_vecs]).float()

    return data
