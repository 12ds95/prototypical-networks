import torch

def k_center_euclidean_dist(x, y):
    # x: N x D
    # y: M x K x D
    n = x.size(0)
    m = y.size(0)
    k = y.size(1)
    d = x.size(1)
    assert d == y.size(2)
    x = x.unsqueeze(1).expand(n, m, d).unsqueeze(2).expand(n, m, k, d)
    y = y.unsqueeze(0).expand(n, m, k, d)
    return torch.pow(x - y, 2).sum(3).sum(2)
