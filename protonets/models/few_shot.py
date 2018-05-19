import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist
from visdom import Visdom
import copy
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
        # norm = z.norm(p=2, dim=1, keepdim=True)
        # z = z.div(norm.expand_as(z).add(1e-20))
        # viz.text("z<br>"+str(z).replace("\n", "<br>"))
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)
        label_predicts = dists.view(n_class, n_query, n_class) 
        
        cosine_similarity_loss = []
        # print("class: %d ###################" % 0)
        for j in range(n_query):
            # print("idx: %d ###################" % j)
            for k in range(j+1, n_query):
                a = label_predicts[0][j][1:].view(1, -1)
                b = label_predicts[0][k][1:].view(1, -1)
                # print(F.cosine_similarity(a, b))                
                cosine_similarity_loss.append(F.cosine_similarity(a, b))
        for i in range(1, n_class-1):
            # print("class: %d ###################" % i)
            for j in range(n_query):
                # print("idx: %d ###################" % j)
                for k in range(j+1, n_query):
                    a = torch.cat((label_predicts[i][j][:i], label_predicts[i][j][i+1:])).view(1, -1)
                    b = torch.cat((label_predicts[i][k][:i], label_predicts[i][k][i+1:])).view(1, -1)
                    # print(F.cosine_similarity(a, b))                
                    cosine_similarity_loss.append(F.cosine_similarity(a, b))
        # print("class: %d ###################" % n_class)
        for j in range(n_query):
            # print("idx: %d ###################" % j)
            for k in range(j+1, n_query):
                a = label_predicts[n_class-1][j][:n_class-1].view(1, -1)
                b = label_predicts[n_class-1][k][:n_class-1].view(1, -1)
                # print(F.cosine_similarity(a, b))                
                cosine_similarity_loss.append(F.cosine_similarity(a, b))
        cosine_similarity_loss = torch.cat(cosine_similarity_loss, 0).mean()
        print(cosine_similarity_loss)  
        log_p_y = F.log_softmax(-dists).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        loss_val = cosine_similarity_loss
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

    return Protonet(encoder)
