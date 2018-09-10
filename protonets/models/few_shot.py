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
        # suppose x is a Variable of size [4, 16], 4 is batch_size, 16 is feature dimension
        # norm = z.norm(p=2, dim=1, keepdim=True)
        # z = z.div(norm.expand_as(z).add(1e-20))
        # viz.text("z<br>"+str(z).replace("\n", "<br>"))
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists).view(n_class, n_query, -1)

        # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        loss = torch.nn.CrossEntropyLoss()
        target = Variable(torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).contiguous().view(-1).long())
        if xq.is_cuda:
            target = target.cuda()
        loss_val = loss((-dists).contiguous(), target)

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.data[0],
            'acc': acc_val.data[0]
        }

class ScaledVarianceRandomNormal(init_ops.Initializer):
    """Initializer that generates tensors with a normal distribution scaled as per https://arxiv.org/pdf/1502.01852.pdf.
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      dtype: The data type. Only floating point types are supported.
    """

    def __init__(self, mean=0.0, factor=1.0, seed=None):
        self.mean = mean
        self.factor = factor
        self.seed = seed

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        if shape:
            n = float(shape[-1])
        else:
            n = 1.0
        for dim in shape[:-2]:
            n *= float(dim)

        self.stddev = np.sqrt(self.factor * 2.0 / n)
        return random_ops.random_uniform(shape, self.mean, self.stddev,
                                        dtype, seed=self.seed)

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.95),
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
    for i in range(4):
        torch.nn.init.kaiming_normal(encoder[i][0].weight, mode='fan_out')
    return Protonet(encoder)

