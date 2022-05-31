from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from mmt.NAS.operations import *
from torch.autograd import Variable
from mmt.NAS.genotypes import PRIMITIVES
from mmt.NAS.genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
      x2 =[]
      for w, op in zip(weights, self._ops):
          x1 = w * op(x)
          x2.append(x1)
      x2 = sum(x2)

      return x2

class Cell(nn.Module):
  def __init__(self, steps, multiplier, C):
    super(Cell, self).__init__()
    self._steps = 1 #1 nodes
    self._multiplier = 1

    self._ops = nn.ModuleList()  #2 operation
    self._bns = nn.ModuleList()
    stride = 1
    op = MixedOp(C, stride)
    self._ops.append(op)

  def forward(self, x, weights):
    s0 = x
    states = [s0]
    offset = 0
    for i in range(self._steps):
        for j, h in enumerate(states):
            s = self._ops[offset + j](h, weights[offset + j])
        offset += len(states)
        states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        # init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        # init.constant(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        # add_block = []
        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck, bias=False)]
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]

        # add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_kaiming)
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block1 = add_block1
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block1(x)
        x1 = self.add_block2(x)
        x2 = self.classifier(x1)
        return x2


class sun_model_search(nn.Module):

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(sun_model_search, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.model.layer4[0].conv2.stride = (1,1)
        self.model.layer4[0].downsample[0].stride = (1,1)

        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveMaxPool2d(1)

        self.cell1 = Cell(1, 1, 256)
        self.cell2 = Cell(1, 1, 512)
        self.cell3 = Cell(1, 1, 1024)
        self.cell4 = Cell(1, 1, 2048)

        self.cells = nn.ModuleList()
        self._steps = 1

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            # out_planes = self.model.fc.in_features
            # # Change the num_features to CNN output channels
            # self.num_features = out_planes
            # self.feat_bn = nn.BatchNorm1d(self.num_features)
            # self.feat_ln = nn.LayerNorm(self.num_features)
            #
            # self.feat_bn.bias.requires_grad_(False)
            # if self.dropout > 0:
            #     self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                #self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                #init.normal_(self.classifier.weight, std=0.001)
                self.classifier = ClassBlock(2048, self.num_classes, num_bottleneck=512)
        # init.constant_(self.feat_bn.weight, 1)
        # init.constant_(self.feat_bn.bias, 0)

        self._initialize_alphas()

    def forward(self, x, T):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)

        weights_cell1 = F.softmax(self.alphas_cell1 / T, dim=-1)
        weights_cell2 = F.softmax(self.alphas_cell2 / T, dim=-1)
        weights_cell3 = F.softmax(self.alphas_cell3 /T, dim=-1)
        weights_cell4 = F.softmax(self.alphas_cell4 / T, dim=-1)


        x = self.cell1(x, weights_cell1)
        x = self.model.layer2(x)
        x = self.cell2(x, weights_cell2)
        x = self.model.layer3(x)
        x = self.cell3(x, weights_cell3)
        x = self.model.layer4(x)
        x = self.cell4(x, weights_cell4)

        x1 = self.gap1(x)
        x2 = self.gap2(x)
        x3 = x1+x2
        x3 = torch.squeeze(x3)

        #bn_x = self.feat_bn(x)

        if self.training is False:
            prob = self.classifier(x3)
            #bn_x = F.normalize(x)
            return x3, prob

        prob = self.classifier(x3)
        return x3, prob

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) )
        num_ops = len(PRIMITIVES)

        self.alphas_cell1 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_cell2 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_cell3 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_cell4 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)

        self._arch_parameters = [
            self.alphas_cell1,
            self.alphas_cell2,
            self.alphas_cell3,
            self.alphas_cell4
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def new(self):
        model_new = sun_model_search( self.depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 1),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:1]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        cell1 = _parse(F.softmax(self.alphas_cell1, dim=-1).data.cpu().numpy())
        cell2 = _parse(F.softmax(self.alphas_cell2, dim=-1).data.cpu().numpy())
        cell3 = _parse(F.softmax(self.alphas_cell3, dim=-1).data.cpu().numpy())
        cell4 = _parse(F.softmax(self.alphas_cell4, dim=-1).data.cpu().numpy())

        #concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            cell1=cell1, cell2=cell2, cell3=cell3, cell4=cell4
        )
        return genotype
