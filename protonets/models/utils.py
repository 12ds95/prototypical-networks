import torch
from visdom import Visdom
viz = Visdom()
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
    dists = torch.exp(-0.5 * (torch.pow(x - y, 2).sum(3)))
    # ret = dists.sum(2)
    ret = torch.max(dists, 2)[0]
    viz.text("dists<br>"+str(dists).replace("\n", "<br>"))
    return ret
