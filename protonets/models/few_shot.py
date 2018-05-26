import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import k_center_euclidean_dist, euclidean_dist

import numpy as np
from sklearn.cluster import KMeans

from visdom import Visdom
from torchvision.utils import make_grid
viz = Visdom()
REG = True
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
        #     break
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

####################################################
        # grad = []
        # weights = []
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         grad.append(torch.zeros(m.weight.size()))
        #         weights.append(torch.zeros(m.weight.size()))
        # i = 0
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         def wrapper(idx):
        #             def extract(var):
        #                 grad[idx] = var
        #             return extract
        #         m.weight.register_hook(wrapper(i))
        #         weights[i] = m.weight
        #         i += 1
        # viz.image(make_grid(weights[0].data,padding=10).numpy())
        # for g in grad:
        #     viz.text(str(g).replace("\n", "<br>"))
        # for w in weights:
        #     viz.text(str(w).replace("\n", "<br>"))
        #     break
####################################################
        # s = []
        # for k, v in params.items():
        #     s.append(str(k)+"<br>")
        # viz.text(s)
####################################################
        # def get_image_input_hook(self, input, output):
        #     viz.text("input: %s, output %s" % (str(input[0].data.size()), str(output.data.size())))
        # self.encoder[0].register_forward_hook(get_image_input_hook)
        # self.encoder[1].register_forward_hook(get_image_input_hook)
        # self.encoder[2].register_forward_hook(get_image_input_hook)
        # self.encoder[3].register_forward_hook(get_image_input_hook)
####################################################        
        # viz.text("z<br>"+str(z).replace("\n", "<br>"))
        zq = z[n_class*n_support:]
        
        # z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        
        # dists = euclidean_dist(zq, z_proto)
        # log_p_y_1 = F.log_softmax(-dists).view(n_class, n_query, -1)
        # viz.text("log_p_y_1<br>"+str(log_p_y_1).replace("\n", "<br>"))
        
        def select_centroids(xs):
            """ 
            Select k centroids for each class
            input: 
            output: (n_class, k_centroid, z_dim)
            """
            # viz.text("xs<br>"+str(xs).replace("\n", "<br>"))
            centroids = None
            nClusters = 2
            for i in range(n_class):
                X = xs[i * n_support:(i+1) * n_support]                
                if nClusters == 2:
                    kmeans = KMeans(n_clusters=nClusters, max_iter=1, random_state=0).fit(X.data.cpu().numpy())
                    c0idxs = np.where(kmeans.labels_ == 0)
                    c1idxs = np.where(kmeans.labels_ == 1)                   
                    if len(c0idxs[0]) == 0:
                        gradCenter = X[c1idxs].contiguous().view(1, len(c1idxs[0]), z_dim).mean(1)
                        if np.array_equal(gradCenter.data.cpu().numpy(), kmeans.cluster_centers_[0]):
                            randCenter = Variable(torch.from_numpy(kmeans.cluster_centers_[1])).view(1, z_dim)
                        else:
                            randCenter = Variable(torch.from_numpy(kmeans.cluster_centers_[0])).view(1, z_dim)
                        if xq.is_cuda:
                            randCenter = randCenter.cuda()
                        kmeans = torch.cat((randCenter, gradCenter), 0) 
                    elif len(c1idxs[0]) == 0:
                        gradCenter = X[c0idxs].contiguous().view(1, len(c0idxs[0]), z_dim).mean(1)                        
                        if np.array_equal(gradCenter.data.cpu().numpy(), kmeans.cluster_centers_[0]):
                            randCenter = Variable(torch.from_numpy(kmeans.cluster_centers_[1])).view(1, z_dim)
                        else:
                            randCenter = Variable(torch.from_numpy(kmeans.cluster_centers_[0])).view(1, z_dim)
                        if xq.is_cuda:
                            randCenter = randCenter.cuda()                        
                        kmeans = torch.cat((randCenter, gradCenter), 0)
                    else:
                        kmeans = torch.cat((X[c0idxs].contiguous().view(1, len(c0idxs[0]), z_dim).mean(1), 
                            X[c1idxs].contiguous().view(1, len(c1idxs[0]), z_dim).mean(1)), 0)
                elif nClusters == 3:
                    kmeans = KMeans(n_clusters=nClusters, max_iter=1, random_state=0).fit(X.data.cpu().numpy())
                    c0idxs = np.where(kmeans.labels_ == 0)
                    c1idxs = np.where(kmeans.labels_ == 1)  
                    c2idxs = np.where(kmeans.labels_ == 2)                  
                    # if len(c0idxs[0]) == 0:
                    #     gradCenter = X[c1idxs].contiguous().view(1, len(c1idxs[0]), z_dim).mean(1)
                    #     if np.array_equal(gradCenter.data.cpu().numpy(), kmeans.cluster_centers_[0]):
                    #         randCenter = Variable(torch.from_numpy(kmeans.cluster_centers_[1])).view(1, z_dim)
                    #     else:
                    #         randCenter = Variable(torch.from_numpy(kmeans.cluster_centers_[0])).view(1, z_dim)
                    #     if xq.is_cuda:
                    #         randCenter = randCenter.cuda()
                    #     kmeans = torch.cat((randCenter, gradCenter), 0) 
                    # elif len(c1idxs[0]) == 0:
                    #     gradCenter = X[c0idxs].contiguous().view(1, len(c0idxs[0]), z_dim).mean(1)                        
                    #     if np.array_equal(gradCenter.data.cpu().numpy(), kmeans.cluster_centers_[0]):
                    #         randCenter = Variable(torch.from_numpy(kmeans.cluster_centers_[1])).view(1, z_dim)
                    #     else:
                    #         randCenter = Variable(torch.from_numpy(kmeans.cluster_centers_[0])).view(1, z_dim)
                    #     if xq.is_cuda:
                    #         randCenter = randCenter.cuda()                        
                    #     kmeans = torch.cat((randCenter, gradCenter), 0)
                    # else:
                    kmeans = torch.cat((X[c0idxs].contiguous().view(1, len(c0idxs[0]), z_dim).mean(1), 
                        X[c1idxs].contiguous().view(1, len(c1idxs[0]), z_dim).mean(1), 
                        X[c2idxs].contiguous().view(1, len(c2idxs[0]), z_dim).mean(1)), 0)
                else:
                    kmeans = KMeans(n_clusters=nClusters, max_iter=1, random_state=0).fit(X.data.cpu().numpy())
                    c0idxs = np.where(kmeans.labels_ == 0)
                    kmeans = X[c0idxs].contiguous().view(1, len(c0idxs[0]), z_dim).mean(1)
                    # kmeans = X.view(1, n_support, z_dim).mean(1)
                if centroids is None:
                    centroids = kmeans.view(1, nClusters, z_dim)
                else:
                    centroids = torch.cat((centroids, kmeans.view(1, nClusters, z_dim)), 0)
            # viz.text("centroid<br>"+str(centroids).replace("\n", "<br>"))
            # if xq.is_cuda:
            #     centroids = centroids.cuda()
            # viz.text("centroids<br>"+str(centroids).replace("\n", "<br>"))
            return centroids
        
        z_proto = select_centroids(z[:n_class*n_support])
        
        dists = k_center_euclidean_dist(zq, z_proto)
        
        #为log_softmax做优化准备: 减去最小数
        #Pick min num in each row (N x M x K) -> (N)  
        minx = torch.min(torch.min(dists.detach(), 1)[0], 1)[0]
        minx = minx.unsqueeze(1).unsqueeze(2).expand(*dists.size())
        # viz.text("minx<br>"+str(minx).replace("\n", "<br>"))
        # viz.text("dists<br>"+str(dists).replace("\n", "<br>"))
        dists = torch.exp(-dists.sub(minx).sum(2))


        # log_p_y_1 = F.log_softmax(-dists).view(n_class, n_query, -1)
        # assert log_p_y_1.size() == log_p_y.size()
####################################################        
        # global REG
        # if REG:
        #     import pickle
        #     with open("z_proto.pkl", "wb") as handle:
        #         pickle.dump(z_proto, handle)
        #     with open("zq.pkl", "wb") as handle:
        #         pickle.dump(zq, handle)           
        #     with open("dists.pkl", "wb") as handle:
        #         pickle.dump((dists.add(1e-27).div(dists.add(1e-27).sum(1).unsqueeze(1).expand(*dists.size()))).view(n_class, n_query, -1), handle)
        #     with open("target_inds.pkl", "wb") as handle:
        #         pickle.dump(target_inds, handle)            
        #     REG = False
        dived = dists.sum(1).unsqueeze(1).expand(*dists.size())
        # nanFilter = 5*1e-20*Variable(torch.ones(dived.size()))
        # if xq.is_cuda:
        #     nanFilter = nanFilter.cuda()
        # dived = torch.max(dived, nanFilter)
        # log_p_y = torch.log((dists.add(1e-20).div(dived)).contiguous().view(n_class, n_query, -1))        
        log_p_y = torch.log((dists.div(dived).add(1e-35)).contiguous().view(n_class, n_query, -1))
        # viz.text("log_p_y<br>"+str(log_p_y).replace("\n", "<br>"))
        # grad = torch.zeros(log_p_y.size())
        # def extract(var):
        #     grad = var
        # log_p_y.register_hook(extract)
        # viz.text(str(log_p_y).replace("\n", "<br>"))
        # viz.text(str(grad).replace("\n", "<br>"))
#################################################### 
        

        # gather像是在某个维度选出一些来
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # viz.text(str(loss_val).replace("\n", "<br>"))
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
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        torch.nn.init.kaiming_normal(block[0].weight)
        return block

    if x_dim[0] == 3:
        encoder = nn.Sequential(
            conv_block(x_dim[0], hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim), 
            conv_block(hid_dim, z_dim),
            Flatten()
        )
    else:
        encoder = nn.Sequential(
            conv_block(x_dim[0], hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten(),
        )

    return Protonet(encoder)
