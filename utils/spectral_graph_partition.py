import scipy
import numpy as np
from sklearn.cluster import KMeans

from utils.data_helper import get_laplacian

__all__ = ['spectral_clustering', 'get_L_cluster_cut']


def spectral_clustering(L, K, seed=1234):
  """
  Implement paper "Shi, J. and Malik, J., 2000. Normalized cuts and image 
  segmentation. IEEE Transactions on pattern analysis and machine intelligence, 
  22(8), pp.888-905."

  Args:
    L: graph Laplacian, numpy or scipy matrix
    K: int, number of clusters

  Returns:
    node_label: list

  N.B.: for simplicity, we only consider simple and undirected graph
  """
  num_nodes = L.shape[0]
  assert (K < num_nodes - 1)

  eig, eig_vec = scipy.sparse.linalg.eigsh(
      L, k=K, which='LM', maxiter=num_nodes * 10000, tol=0, mode='normal')
  kmeans = KMeans(n_clusters=K, random_state=seed).fit(eig_vec.real)

  return kmeans.labels_


def get_L_cluster_cut(L, node_label):
  adj = L - np.diag(np.diag(L))
  adj[adj != 0] = 1.0
  num_nodes = adj.shape[0]
  idx_row, idx_col = np.meshgrid(range(num_nodes), range(num_nodes))
  idx_row, idx_col = idx_row.flatten().astype(
      np.int64), idx_col.flatten().astype(np.int64)
  mask = (node_label[idx_row] == node_label[idx_col]).reshape(
      num_nodes, num_nodes).astype(np.float)

  adj_cluster = adj * mask
  adj_cut = adj - adj_cluster
  L_cut = get_laplacian(adj_cut, graph_laplacian_type='L4')
  L_cluster = get_laplacian(adj_cluster, graph_laplacian_type='L4')

  return L_cluster, L_cut