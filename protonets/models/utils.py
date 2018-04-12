import torch
from visdom import Visdom
viz = Visdom()
REG = True
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
    # viz.text("x_k<br>"+str(x).replace("\n", "<br>"))
    # viz.text("y_k<br>"+str(y).replace("\n", "<br>"))
    # dists = torch.pow(x - y, 2).sum(3)
    # viz.text("dists_k<br>"+str(dists).replace("\n", "<br>"))
    
    dists = torch.pow(x - y, 2).sum(3)
    # viz.text("dists_k<br>"+str(dists).replace("\n", "<br>"))
    # ret = torch.max(dists, 2)[0]
    return dists

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    # viz.text("x<br>"+str(x).replace("\n", "<br>"))
    # viz.text("y<br>"+str(y).replace("\n", "<br>"))
    dists = torch.pow(x - y, 2).sum(2)
    # viz.text("dists<br>"+str(dists).replace("\n", "<br>"))
    return torch.pow(x - y, 2).sum(2)