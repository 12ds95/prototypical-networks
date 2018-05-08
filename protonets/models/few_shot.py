import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist
from visdom import Visdom
import copy

viz = Visdom()

def attentionMetric(x, y, learnedMetric):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)
        assert d == learnedMetric.size(0)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        # d: N x M x D -> N x M
        d = (x - y).view(n*m, d)
        # let K = NxM,
        # K x 1 x D bmm K x D x 1 => K x 1 x 1
        # viz.text("metric<br>"+str(self.learnedMetric).replace("\n", "<br>")) 
        return d.mm(learnedMetric).unsqueeze(1).bmm(d.unsqueeze(1).transpose(1, 2)).view(n, m)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, shared_layers, win_size, attention, n_corase, fine_encoders):
        super(Protonet, self).__init__()
        # self.register_buffer('shared_layers', shared_layers)
        self.win_size = win_size
        self.shared_layers = shared_layers
        self.attention = attention
        self.n_corase = n_corase
        for i in range(self.n_corase):
            self.add_module('fine_encoder_'+str(i), fine_encoders[i])

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

        # share layers part
        z_share = self.shared_layers.forward(x)

        # attention matrix part
        z_attention = self.attention.forward(z_share)
        diag_vector = F.softmax(z_attention[:n_class*n_support].mean(0), dim=0).unsqueeze(1).unsqueeze(2).expand(
            z_attention.size()[1], self.win_size, self.win_size).contiguous().view(-1) * z_attention.size()[1]
        # fine feature part
        z = self._modules['fine_encoder_0'].forward(z_share)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = attentionMetric(zq, z_proto, torch.diag(diag_vector))

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        # loss_diff = []
        # for i in range(self.n_corase):
        #     for j in range(i, self.n_corase):
        #         z1 = self._modules['fine_encoder_'+str(i)][0][0].weight
        #         z2 = self._modules['fine_encoder_'+str(j)][0][0].weight
        #         loss_diff.append(1 - torch.pow(z1 - z2, 2))
        # weight = 1e-5
        # if loss_diff != []:
        #     loss_val = loss_val + weight  / len(loss_diff) * torch.sum(torch.cat(loss_diff, 0))

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        
        return loss_val, {
            'loss': loss_val.data[0],
            'acc': acc_val.data[0]
        }

        # return loss_val_corase, {
        #     'loss': loss_val_corase.data[0],
        #     'acc': acc_val_corase.data[0]
        # }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    n_corase = kwargs['n_corase']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    shared_layers = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
    )

    # model = torch.load('proto_results/m30_5way5shot/best_model.t7')

    # load pretrained layers 
    # shared_layers = nn.Sequential(
    #     copy.deepcopy(model.encoder[0]),
    #     copy.deepcopy(model.encoder[1]),
    #     copy.deepcopy(model.encoder[2])
    # )

    def gap_block(in_channels, out_channels, pre_size):
        return nn.Sequential(
            nn.Conv2d(hid_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(pre_size),
        )

    attention = nn.Sequential(
        conv_block(hid_dim, hid_dim),
        gap_block(hid_dim, hid_dim, x_dim[1] // 16),
        Flatten()
    )

    fine_encoders = []
    for i in range(n_corase):
        fine_encoders.append(nn.Sequential(
                    conv_block(hid_dim, z_dim),
                    Flatten()
        ))
    
    n_2x2_MaxPool = 4
    win_size = x_dim[1] // pow(2, n_2x2_MaxPool)

    return Protonet(shared_layers, win_size, attention, n_corase, fine_encoders)
