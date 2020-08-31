import torch

def similarity_q(z: torch.Tensor, u: torch.Tensor):
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