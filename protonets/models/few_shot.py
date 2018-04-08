import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import k_center_euclidean_dist

import numpy as np
from sklearn.cluster import KMeans

from visdom import Visdom
from torchvision.utils import make_grid
viz = Visdom()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        
        self.encoder = encoder

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query
        # for cs in xs:
        #     viz.image(make_grid(cs.data,padding=10).numpy())
        # viz.text("support " + str(xs.size()) + "query " +  str(xq.size()))
        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        #这里不是很懂....target -> label转成idx？？
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)
        viz.text("z<br>"+str(z).replace("\n", "<br>"))
        # z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        def select_centroids(xs):
            """ 
            Select k centroids for each class
            input: 
            output: (n_class, k_centroid, z_dim)
            """
            viz.text("xs<br>"+str(xs).replace("\n", "<br>"))
            centroids = []
            for i in range(n_class):
                X = xs[i * n_support:(i+1) * n_support].data.cpu().numpy()
                if n_support // 2:
                    kmeans = KMeans(n_clusters=2, max_iter=1, random_state=0).fit(X).cluster_centers_
                else:
                    kmeans = [X]
                centroids.append(kmeans)
            centroids = Variable(torch.from_numpy(np.array(centroids)))
            # viz.text("centroid<br>"+str(centroids).replace("\n", "<br>"))
            if xq.is_cuda:
                centroids = centroids.cuda()
            return centroids
        
        z_proto = select_centroids(z[:n_class*n_support])
        zq = z[n_class*n_support:]
        dists = k_center_euclidean_dist(zq, z_proto)

        #log_p_y_1 = F.log_softmax(-dists).view(n_class, n_query, -1)
        #viz.text(str((dists.div(dists.sum(1).unsqueeze(1).expand(*dists.size()))).view(n_class, n_query, -1)))       
        log_p_y = torch.log((dists.div(dists.sum(1).unsqueeze(1).expand(*dists.size()))).view(n_class, n_query, -1))
        #assert log_p_y_1.size() == log_p_y.size()

        #gather像是在某个维度选出一些来
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.data[0],
            'acc': acc_val.data[0]
        }

#这个register是怎么工作的，不懂？？？
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

    return Protonet(encoder)
