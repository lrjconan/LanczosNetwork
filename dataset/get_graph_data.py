import os
import glob
import pickle
import numpy as np
import networkx as nx
from utils.data_helper import get_multi_graph_laplacian_eigs, get_graph_laplacian_eigs, get_laplacian

save_dir = '../data/synthetic/'

if not os.path.exists(save_dir):
  os.mkdir(save_dir)
  print('made directory {}'.format(save_dir))


def gen_data(min_num_nodes=20,
             max_num_nodes=100,
             num_graphs=10,
             node_emb_dim=10,
             graph_emb_dim=2,
             edge_prob=0.5,
             seed=123):
  """
    Generate synthetic graph data for graph regression, i.e., given node 
    embedding and graph structure as input, predict a graph embedding 
    as output.

    N.B.: modification to other tasks like node classification is straightforward

    A single graph in your dataset should contin:
      X: Node embedding, numpy array, shape N X D where N is # nodes
      A: Graph structure, numpy array, shape N X N X E where E is # edge types
      Y: Graph embedding, numpy array, shape N X O
  """
  npr = np.random.RandomState(seed)
  N = npr.randint(min_num_nodes, high=max_num_nodes+1, size=num_graphs)

  data = []
  for ii in range(num_graphs):    
    data_dict = {}
    data_dict['X'] = npr.randn(N[ii], node_emb_dim)
    # we assume # edge type = 1, but you can easily extend it to be more than 1
    data_dict['A'] = np.expand_dims(
        nx.to_numpy_matrix(
            nx.fast_gnp_random_graph(N[ii], edge_prob, seed=npr.randint(1000))),
        axis=2)
    data_dict['Y'] = npr.randn(1, graph_emb_dim)
    data += [data_dict]

  return data


def dump_data(data_list, tag='train'):
  count = 0
  print('Dump {} data!'.format(tag))
  for data in data_list:

    data_dict = {}
    data_dict['node_feat'] = data['X']
    adjs = data['A']
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
    data_dict['label'] = data['Y']

    pickle.dump(
        data_dict,
        open(
            os.path.join(save_dir, 'synthetic_{}_{:07d}.p'.format(tag, count)),
            'wb'))

    count += 1

  print('100.0 %%')


if __name__ == '__main__':
  train_dataset = gen_data(seed=123)
  dev_dataset = gen_data(seed=456)
  test_dataset = gen_data(seed=789)

  dump_data(train_dataset, 'train')
  dump_data(dev_dataset, 'dev')
  dump_data(test_dataset, 'test')
