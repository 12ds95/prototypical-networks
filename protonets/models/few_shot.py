import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist
from visdom import Visdom
viz = Visdom()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder, learnedMetric):
        super(Protonet, self).__init__()
        
        self.encoder = encoder
        self.learnedMetric = learnedMetric

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        # viz.text("z<br>"+str(z).replace("\n", "<br>"))
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)
        def learnedMetric(x, y):
            # x: N x D
            # y: M x D
            n = x.size(0)
            m = y.size(0)
            d = x.size(1)
            assert d == y.size(1)
            assert d == self.learnedMetric.size(0)

            x = x.unsqueeze(1).expand(n, m, d)
            y = y.unsqueeze(0).expand(n, m, d)
            # d: N x M x D -> N x M
            d = (x - y).view(n*m, d)
            # let K = NxM,
            # K x 1 x D bmm K x D x 1 => K x 1 x 1
            viz.image(self.learnedMetric) 
            return d.mm(self.learnedMetric).unsqueeze(1).bmm(d.unsqueeze(1).transpose(1, 2)).view(n, m)
        
        dists = learnedMetric(zq, z_proto)
        log_p_y = F.log_softmax(-dists).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.data[0],
            'acc': acc_val.data[0]
        }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )
    n_2x2_MaxPool = 4
    w = pow(x_dim[1] // pow(2, n_2x2_MaxPool), 2) * z_dim
    # print(w)
    learnedMetric = nn.Parameter(torch.rand(w, w))
    return Protonet(encoder, learnedMetric)