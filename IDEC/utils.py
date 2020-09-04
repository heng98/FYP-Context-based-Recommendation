import torch
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment


def soft_assign(z: torch.Tensor, u: torch.Tensor):
    """
    Similarity between embedded point z_i and cluster center u_j
    Assume alpha = 1 as stated in paper
    q_ij = 1/(1+dist(z_i, u_j)^2), then normalize it
    """
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - u, 2), dim=2))
    q = q.t() / torch.sum(q, 1)
    q = q.t()

    return q

def target_dist(q):
    p = torch.pow(q, 2) / torch.sum(q, 0)
    p = p.t() / torch.sum(p, 1)
    p = p.t()

    return p

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """

    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = w[row_ind, col_ind].sum() / y_pred.size
    
    return reassignment, accuracy