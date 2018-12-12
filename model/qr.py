import torch


def QR_gram_schmidt(A):
  m = A.shape[0]
  n = A.shape[0]
  Q = torch.zeros_like(A)
  R = torch.zeros(n, n)
  if A.is_cuda:
    R = R.cuda()
  
  for ii in range(n):
    if ii > 1:            
      R[:ii, ii] = Q[:, :ii].t().mm(A[:, ii].view(-1, 1)).view(-1)
      v = A[:, ii].view(-1, 1) - Q[:, :ii].mm(R[:ii, ii].unsqueeze(dim=1))
    else:      
      R[ii, ii] = (Q[:, ii] * A[:, ii]).sum()
      v = A[:, ii] - Q[:, ii] * R[ii, ii]
    
    R[ii, ii] = torch.norm(v)
    Q[:, ii] = v.view(-1) / R[ii, ii]

  return Q, R


def QR_householder(A):
  m = A.shape[0]
  Q = torch.eye(m)
  if A.is_cuda:
    Q = Q.cuda()

  for ii in range(m):
    ee = torch.zeros_like(A[:, ii])
    ee[ii] = A[:, ii].norm() * torch.sign(A[ii, ii])
    mask = torch.zeros_like(A[:, ii])
    mask[ii,:] = 1.0
    w = A[:, ii] * mask + ee
    v = w / w.norm()
    H = torch.eye(m)
    if A.is_cuda:
      H = H.cuda()
    H = H - 2.0 * v.mm(v.T())
    A = H.mm(A)
    Q = H.mm(Q)  

  R = A
  return Q, R


def eig_QR(A, max_iter=100):
  T = A
  U = torch.eye(A.shape[0])

  for ii in range(max_iter):
    Q, R = QR_gram_schmidt(T)
    T = R.mm(Q)
    U = U.mm(Q)

  return T, U

def test():
  A = torch.randn(5, 5)
  A = A + A.t()
  Q, R = QR_gram_schmidt(A)
  Q_gt, R_gt = torch.qr(A)
  T, U = eig_QR(A)
  D, V = torch.symeig(A)
  import pdb; pdb.set_trace()

if __name__ == '__main__':
  test()
