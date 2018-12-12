import numpy as np
from scipy import sparse as sp

EPS = float(np.finfo(np.float32).eps)
DRAW_HISTOGRAM = False
DRAW_APPROXIMATION = False


def check_dist(dist):
  for dd in dist:
    if not isinstance(dd, int) and dd != 'inf':
      raise ValueError("Non-supported value of diffusion distance")

  return dist


def read_idx_file(file_name):
  idx = []
  with open(file_name) as f:
    for line in f:
      idx += [int(line)]

  return idx


def normalize_features(mx):
  """Row-normalize sparse matrix"""
  rowsum = np.array(mx.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv)
  mx = r_mat_inv.dot(mx)
  return mx


def check_symmetric(m, tol=1e-8):

  if sp.issparse(m):
    if m.shape[0] != m.shape[1]:
      raise ValueError('m must be a square matrix')

    if not isinstance(m, sp.coo_matrix):
      m = sp.coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
      return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    return np.allclose(vl, vu, atol=tol)
  else:
    return np.allclose(m, m.T, atol=tol)


def preprocess_feature(feature, norm_method=None):
  """ Normalize feature matrix """

  if norm_method == 'L1':
    # L1 norm
    feature /= (feature.sum(1, keepdims=True) + EPS)

  elif norm_method == 'L2':
    # L2 norm
    feature /= (np.sqrt(np.square(feature).sum(1, keepdims=True)) + EPS)

  elif norm_method == 'std':
    # Standardize
    std = np.std(feature, axis=0, keepdims=True)
    feature -= np.mean(feature, 0, keepdims=True)
    feature /= (std + EPS)
  else:
    # nothing
    pass

  return feature


def normalize_adj(A, is_sym=True, exponent=0.5):
  """
    Normalize adjacency matrix

    is_sym=True: D^{-1/2} A D^{-1/2}
    is_sym=False: D^{-1} A
  """
  rowsum = np.array(A.sum(1))

  if is_sym:
    r_inv = np.power(rowsum, -exponent).flatten()
  else:
    r_inv = np.power(rowsum, -1.0).flatten()

  r_inv[np.isinf(r_inv)] = 0.

  if sp.isspmatrix(A):
    r_mat_inv = sp.diags(r_inv.squeeze())
  else:
    r_mat_inv = np.diag(r_inv)

  if is_sym:
    return r_mat_inv.dot(A).dot(r_mat_inv)
  else:
    return r_mat_inv.dot(A)


def get_laplacian(adj, graph_laplacian_type='L1', alpha=0.5):
  """
    Compute Graph Laplacian

    Args:
      adj: shape N X N, adjacency matrix, could be numpy or scipy sparse array
      graph_laplacian_type:
        -L1: use combinatorial graph Laplacian, L = D - A
        -L2: use symmetric graph Laplacian, L = I - D^{-1/2} A D^{-1/2}
        -L3: use asymmetric graph Laplacian, L = I - D^{-1} A
        -L4: use symmetric GCN renormalization trick, L = D^{-1/2} ( I + A ) D^{-1/2}
        -L5: use asymmetric GCN renormalization trick, L = D^{-1} ( I + A )
        -L6: use symmetric diffusion map, L = D^{-alpha} A D^{-alpha},
              where e.g., A_{i,j} = k(x_i, x_j), alpha is typically from [0, 1]
        -L7: use asymmetric diffusion map, L = D^{-1} A, A as L5

    Returns:
      L: shape N X N, graph Laplacian matrix
  """

  assert len(adj.shape) == 2 and adj.shape[0] == adj.shape[1]

  if sp.isspmatrix(adj):
    identity_mat = sp.eye(adj.shape[0])
  else:
    identity_mat = np.eye(adj.shape[0])

  if graph_laplacian_type == 'L1':
    if sp.isspmatrix(adj):
      L = sp.diags(np.array(adj.sum(axis=1)).squeeze()) - adj
    else:
      L = np.diag(adj.sum(axis=1).squeeze()) - adj
  elif graph_laplacian_type == 'L2':
    L = identity_mat - normalize_adj(adj, is_sym=True)
  elif graph_laplacian_type == 'L3':
    L = identity_mat - normalize_adj(adj, is_sym=False)
  elif graph_laplacian_type == 'L4':
    L = normalize_adj(identity_mat + adj, is_sym=True)
  elif graph_laplacian_type == 'L5':
    L = normalize_adj(identity_mat + adj, is_sym=False)
  elif graph_laplacian_type == 'L6':
    L = normalize_adj(adj, is_sym=True, exponent=alpha)
  elif graph_laplacian_type == 'L7':
    L = normalize_adj(adj, is_sym=False)
  else:
    raise ValueError('Unsupported Graph Laplacian!')

  return L


def get_graph_laplacian_eigs(adj,
                             k=100,
                             graph_laplacian_type='L1',
                             alpha=0.5,
                             use_eigen_decomp=False,
                             is_sym=True):
  """
    Compute first k largest eigenvalues and eigenvectors of graph Laplacian

    Args:
      adj: shape N X N, adjacency matrix, could be numpy or scipy sparse array
      k: int, number of eigenvalues and eigenvectors
      graph_laplacian_type: see arguments of get_laplacian for explanation
      alpha: float, scale parameter of diffusion map
      use_eigen_decomp: bool, indicates whether use eigen decomposition, note
                        it is computationally heavy for large size adj
      is_sym: bool, indicates whether adj is symmetric

    Returns:
      eig: shape K X 1, eigenvalues
      V: shape N X K, eigenvectors, each column is an eigenvector
      L: shape N X N, graph Laplacian, has the same type as adj
  """

  # compute symmetric graph Laplacian
  L = get_laplacian(adj, graph_laplacian_type=graph_laplacian_type, alpha=alpha)
  assert is_sym == check_symmetric(L)

  try:
    # apply Eigen-Decomposition methods
    if use_eigen_decomp:
      if is_sym:
        eigs, V = np.linalg.eigh(L)
      else:
        eigs, V = np.linalg.eig(L)
    else:
      if is_sym:
        # apply Lanczos
        # N.B.: it can be very slow if k >= 2000
        eigs, V = sp.linalg.eigsh(L, k=k, which='LM')
      else:
        # apply Arnoldi
        eigs, V = sp.linalg.eigs(L, k=k, which='LM')

    ### TODO: handle cases where eigs are complex
    if np.any(np.iscomplex(eigs)):
      print('Warning: there are complex eigenvalues!')

    # magnitude
    eigs_M = np.abs(eigs)

    # sort it following descending order
    idx = np.argsort(-eigs_M, kind='mergesort')
    eigs = eigs[idx[:k]]
    V = V[:, idx[:k]]
  except:
    print('Warning: computing eigenvalues failed!')
    eigs, V = None, None

  # draw some statistics
  if DRAW_HISTOGRAM:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)
    plt.figure()
    sns.distplot(eigs, bins=100, kde=False, rug=False)
    plt.savefig('hist_eigs.png', bbox_inches='tight')

  if DRAW_APPROXIMATION:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    rank_set = [5, 10, 20, 50, 100, 200, 300]
    loss = []
    L1 = V.dot(np.diag(eigs)).dot(V.T)

    for k in rank_set:
      eigk, Vk = sp.linalg.eigsh(L, k=k)
      Lk = Vk.dot(np.diag(eigk)).dot(Vk.T)
      loss += [np.linalg.norm(L1 - Lk)]

    plt.figure()
    sns.tsplot(loss)
    plt.xticks(np.arange(len(loss)), rank_set)
    plt.savefig('loss_eigs.png', bbox_inches='tight')

  return eigs, V, L


def get_multi_graph_laplacian_eigs(adjs,
                                   k=100,
                                   graph_laplacian_type='L1',
                                   alpha=0.5,
                                   use_eigen_decomp=False,
                                   is_sym=True):
  """
    See comments of get_graph_laplacian_eigs for more information

    Args:
      adjs: shape N X N X K, K is # edge types

    Returns:
      eigs_list: list of eigenvalues for each edge type
      V_list: list of eigenvectors for each edge type
      L_list: list of graph Laplacian for each edge type
  """
  V_list = []
  L_list = []
  eigs_list = []
  for ii in range(adjs.shape[2]):
    eigs, V, L = get_graph_laplacian_eigs(
        adjs[:, :, ii],
        k=k,
        graph_laplacian_type=graph_laplacian_type,
        alpha=alpha,
        use_eigen_decomp=use_eigen_decomp,
        is_sym=is_sym)
    V_list += [V]
    L_list += [L]
    eigs_list += [eigs]

  return eigs_list, V_list, L_list


def test_laplacian():
  adj = np.array([[0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0]]).astype(np.float32)
  print('=' * 80)
  print('L1 = {}'.format(
      get_laplacian(adj, graph_laplacian_type='L1', alpha=0.5)))
  print('=' * 80)
  print('L2 = {}'.format(
      get_laplacian(adj, graph_laplacian_type='L2', alpha=0.5)))
  print('=' * 80)
  print('L3 = {}'.format(
      get_laplacian(adj, graph_laplacian_type='L3', alpha=0.5)))
  print('=' * 80)
  print('L4 = {}'.format(
      get_laplacian(adj, graph_laplacian_type='L4', alpha=0.5)))
  print('=' * 80)
  print('L5 = {}'.format(
      get_laplacian(adj, graph_laplacian_type='L5', alpha=0.5)))
  print('=' * 80)
  print('L6 = {}'.format(
      get_laplacian(adj, graph_laplacian_type='L6', alpha=0.5)))
  print('=' * 80)
  print('L7 = {}'.format(
      get_laplacian(adj, graph_laplacian_type='L7', alpha=0.5)))

  adj_sp = sp.coo_matrix(adj)
  print('=' * 80)
  print('L1_sp = {}'.format(
      get_laplacian(adj_sp, graph_laplacian_type='L1', alpha=0.5)))
  print('=' * 80)
  print('L2_sp = {}'.format(
      get_laplacian(adj_sp, graph_laplacian_type='L2', alpha=0.5)))
  print('=' * 80)
  print('L3_sp = {}'.format(
      get_laplacian(adj_sp, graph_laplacian_type='L3', alpha=0.5)))
  print('=' * 80)
  print('L4_sp = {}'.format(
      get_laplacian(adj_sp, graph_laplacian_type='L4', alpha=0.5)))
  print('=' * 80)
  print('L5_sp = {}'.format(
      get_laplacian(adj_sp, graph_laplacian_type='L5', alpha=0.5)))
  print('=' * 80)
  print('L6_sp = {}'.format(
      get_laplacian(adj_sp, graph_laplacian_type='L6', alpha=0.5)))
  print('=' * 80)
  print('L7_sp = {}'.format(
      get_laplacian(adj_sp, graph_laplacian_type='L7', alpha=0.5)))


def test_eigs():
  adj = np.array([[0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0]]).astype(np.float32)

  D, V, L = get_graph_laplacian_eigs(
      adj,
      k=100,
      graph_laplacian_type='L1',
      alpha=0.5,
      use_eigen_decomp=True,
      is_sym=True)
  print('=' * 80)
  print('Eig values = {}'.format(D))
  print('=' * 80)
  print('Eig vectors = {}'.format(V))
  print('=' * 80)
  print('Graph Laplacian = {}'.format(L))

  adj_sp = sp.coo_matrix(adj)
  D_sp, V_sp, L_sp = get_graph_laplacian_eigs(
      adj_sp,
      k=100,
      graph_laplacian_type='L1',
      alpha=0.5,
      use_eigen_decomp=True,
      is_sym=True)
  print('=' * 80)
  print('Eig values = {}'.format(D_sp))
  print('=' * 80)
  print('Eig vectors = {}'.format(V_sp))
  print('=' * 80)
  print('Graph Laplacian = {}'.format(L_sp))


if __name__ == '__main__':
  # test_laplacian()
  test_eigs()
