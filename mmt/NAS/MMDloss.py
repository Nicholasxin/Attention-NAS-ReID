import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable


class hetero_loss(nn.Module):
    def __init__(self, margin=0.1, dist_type='l2'):
        super(hetero_loss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, label1, label2):
        feat_size = feat1.size()[1]
        feat_num = feat1.size()[0]
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist = max(0, self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, self.dist(center1, center2) - self.margin)
            elif self.dist_type == 'cos':
                if i == 0:
                    dist = max(0, 1 - self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, 1 - self.dist(center1, center2) - self.margin)

        return dist


class Center_TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(Center_TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.dist = self.dist = nn.MSELoss(reduction='sum')

    def forward(self, input1, input2, targets1, targets2):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - inputs1: visible , inputs2: thermal
        - targets: ground truth labels with shape (num_classes)
        """
        feat_size = input1.size()[1]
        label_num_c = len(targets1.unique())
        label_num_all = len(targets1)
        input1_all = input1.chunk(label_num_all, 0)
        input2_all = input2.chunk(label_num_all, 0)
        input1_c = input1.chunk(label_num_c, 0)
        input2_c = input2.chunk(label_num_c, 0)
        center1 = []
        center2 = []
        for i in range(label_num_c):
            center11 = torch.mean(input1_c[i], dim=0)
            center22 = torch.mean(input2_c[i], dim=0)
            center1.append(center11)
            center2.append(center22)
        dist1 = 0
        dist2 = 0
        # center1 = torch.mean(input1, dim=0)
        for i in range(label_num_c):
            for j in range(label_num_c):
                dist = max(0, self.dist(center1[i], center2[i]) - self.dist(center1[i], center2[j]))
                if dist > dist1:
                    dist1 = dist
                else:
                    dist1 = dist1
            dist2 += dist1

        # dist = dist/label_num_c

        return dist


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    ½«Ô´ÓòÊý¾ÝºÍÄ¿±êÓòÊý¾Ý×ª»¯ÎªºË¾ØÕó£¬¼´ÉÏÎÄÖÐµÄK
    Params:
	    source:
	    target:
	    kernel_mul:
	    kernel_num:
	    fix_sigma:
		sum(kernel_val):
     '''
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(input1, input2, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(input1.size()[0])

    kernels = guassian_kernel(input1, input2,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class center_loss(nn.Module):
    def __init__(self, margin=0.1, dist_type='l2'):
        super(center_loss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, feat3):

        center1 = torch.mean(feat1, dim=0)
        center2 = torch.mean(feat2, dim=0)
        center3 = torch.mean(feat3, dim=0)
        dist1 = max(0, self.dist(center1, center2))
        dist2 = max(0, self.dist(center3, center2))
        dist = dist1 + dist2
        return dist


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class center_loss1(nn.Module):
    def __init__(self, margin=0.1, dist_type='l2'):
        super(center_loss1, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2):

        feat11 = feat1.chunk(4, 0)
        feat22 = feat2.chunk(4, 0)

        mix1 = torch.cat((feat11[0], feat22[0]), 0)
        mix2 = torch.cat((feat11[1], feat22[1]), 0)
        mix3 = torch.cat((feat11[2], feat22[2]), 0)
        mix4 = torch.cat((feat11[3], feat22[3]), 0)
        mix = torch.cat((mix1, mix2), 0)
        mix = torch.cat((mix, mix3), 0)
        mix = torch.cat((mix, mix4), 0)
        mix = mix.chunk(8, 0)

        center = []
        center1 = torch.mean(mix1, dim=0)
        # center1 = center1.view(-1, 512)
        center2 = torch.mean(mix2, dim=0)
        # center2 = center2.view(-1, 512)
        center3 = torch.mean(mix3, dim=0)
        # center3 = center3.view(-1, 512)
        center4 = torch.mean(mix4, dim=0)
        # center4 = center4.view(-1, 512)
        center.append(center1)
        center.append(center2)
        center.append(center3)
        center.append(center4)

        dist_all = 0
        for i in range(4):
            mean1 = torch.mean(mix[2 * i], dim=0)
            mean2 = torch.mean(mix[2 * i + 1], dim=0)
            dist1 = self.dist(mean1, center[i])
            dist2 = self.dist(mean2, center[i])
            dist = dist1 + dist2
            dist_all += dist

        return dist_all / 2