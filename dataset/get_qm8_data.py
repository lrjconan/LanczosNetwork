import os
import glob
import pickle
import numpy as np
import deepchem as dc
from utils.data_helper import get_multi_graph_laplacian_eigs, get_graph_laplacian_eigs, get_laplacian

global num_nodes, num_edges, num_graphs 
num_nodes = 0
num_edges = 0
num_graphs = 0
n_atom_feat = 70
n_pair_feat = 6  # edge type #
data_folder = '../data/QM8/'
save_dir = '../data/QM8/preprocess'

if not os.path.exists(data_folder):
  os.mkdir(data_folder)
  print('made directory {}'.format(data_folder))

if not os.path.exists(save_dir):
  os.mkdir(save_dir)
  print('made directory {}'.format(save_dir))


def to_graph(mol):
  # number of atoms in each molecule
  n_atoms = mol.get_num_atoms()

  # atom features
  atom_feat = mol.get_atom_features()

  # convert 1-of-K encoding to index
  atom_feat = np.argmax(atom_feat, axis=1)

  # edge features, shape N X N X 8
  # first 6 channels: bond-wise adjacency matrix
  # 6-th channel: is-in-the-same-ring adjacency matrix
  # 7-th channel: distance matrix
  pair_feat = mol.get_pair_features()

  return atom_feat, pair_feat[:, :, :6]


def dump_data(dataset, tag='train'):
  count = 0
  global num_graphs
  print('Dump {} data!'.format(tag))
  for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
      batch_size=1, deterministic=True, pad_batches=False):
    assert len(X_b) == 1

    if count % 100 == 0:
      print('{:.1f} %%\r'.format(100 * count / float(len(dataset))), end='')

    data_dict = {}
    for im, mol in enumerate(X_b):
      global num_nodes, num_edges
      node_feat, adjs = to_graph(mol)
      data_dict['node_feat'] = node_feat

      adj_simple = np.sum(adjs, axis=2)
      D_list, V_list, L_list = get_multi_graph_laplacian_eigs(
          adjs, graph_laplacian_type='L4', use_eigen_decomp=True, is_sym=True)
      D, V, L4 = get_graph_laplacian_eigs(
          adj_simple,
          graph_laplacian_type='L4',
          use_eigen_decomp=True,
          is_sym=True)

      L6 = get_laplacian(adj_simple, graph_laplacian_type='L6')
      L7 = get_laplacian(adj_simple, graph_laplacian_type='L7')

      data_dict['L_multi'] = np.stack(L_list, axis=2)
      data_dict['L_simple_4'] = L4
      data_dict['L_simple_6'] = L6
      data_dict['L_simple_7'] = L7

      # N.B.: for some edge type, adjacency matrix may be diagonal
      data_dict['D_simple'] = D if D is not None else np.ones(adjs.shape[0])
      data_dict['V_simple'] = V if V is not None else np.eye(adjs.shape[0])
      data_dict['D_multi'] = D_list
      data_dict['V_multi'] = V_list
      
      num_nodes += node_feat.shape[0]
      num_edges += np.sum(adj_simple) / 2.0
    
    num_graphs += 1.0
    data_dict['label'] = y_b
    data_dict['label_weight'] = w_b

    pickle.dump(
        data_dict,
        open(
            os.path.join(save_dir, 'QM8_preprocess_{}_{:07d}.p'.format(
                tag, count)), 'wb'))

    count += 1

  print('100.0 %%')


if __name__ == '__main__':
  tasks, datasets, transformers = dc.molnet.load_qm8(
      featurizer='MP', reload=False)
  train_dataset, dev_dataset, test_dataset = datasets

  dump_data(train_dataset, 'train')
  dump_data(dev_dataset, 'dev')
  dump_data(test_dataset, 'test')

  mean_label = transformers[0].y_means
  std_label = transformers[0].y_stds

  print('mean = {}'.format(mean_label))
  print('std = {}'.format(std_label))
  print('average nodes per graph = {}'.format(num_nodes / num_graphs))
  print('average edges per graph = {}'.format(num_edges / num_graphs))

  meta_data = {'mean': mean_label, 'std': std_label}
  pickle.dump(meta_data, open(os.path.join(data_folder, 'QM8_meta.p'), 'wb'))
