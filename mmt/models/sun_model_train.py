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
from mmt.NAS.cnnutils import drop_path


class Cell1(nn.Module):

    def __init__(self, genotype, C):
        super(Cell1, self).__init__()
        print( C)
        cell1, indices1 = zip(*genotype.cell1)

        self._compile(C, cell1, indices1)

    def _compile(self, C, cell1, indices1):
        assert len(cell1) == len(indices1)
        self._steps = 1
        self.multiplier = 1

        self._ops = nn.ModuleList()
        for name, index in zip(cell1, indices1):
            op = OPS[name](C, 1, True)
            self._ops += [op]
        self._indices = indices1

    def forward(self, x):
        s0 = x
        states = [s0]
        for i in range(self._steps):
            h1 = states[self._indices[ i]]
            op1 = self._ops[i]
            h1 = op1(h1)
            s = h1
            states += [s]
        return s

class Cell2(nn.Module):

    def __init__(self, genotype, C):
        super(Cell2, self).__init__()
        print( C)
        cell2, indices2 = zip(*genotype.cell2)
        self._compile(C, cell2, indices2)

    def _compile(self, C, cell2, indices2):
        assert len(cell2) == len(indices2)
        self._steps = 1
        self.multiplier = 1
        self._ops = nn.ModuleList()
        for name, index in zip(cell2, indices2):
            op = OPS[name](C, 1, True)
            self._ops += [op]
        self._indices = indices2

    def forward(self, x):
        s0 = x
        states = [s0]
        for i in range(self._steps):
            h1 = states[self._indices[ i]]
            op1 = self._ops[i]
            h1 = op1(h1)
            s = h1
            states += [s]
        return s

class Cell3(nn.Module):

    def __init__(self, genotype, C):
        super(Cell3, self).__init__()
        print( C)

        cell3, indices3 = zip(*genotype.cell3)
        self._compile(C, cell3, indices3)

    def _compile(self, C, cell3, indices3):
        assert len(cell3) == len(indices3)
        self._steps = 1
        self.multiplier = 1

        self._ops = nn.ModuleList()
        for name, index in zip(cell3, indices3):
            op = OPS[name](C, 1, True)
            self._ops += [op]
        self._indices = indices3

    def forward(self, x):
        s0 = x
        states = [s0]
        for i in range(self._steps):
            h1 = states[self._indices[ i]]
            op1 = self._ops[i]
            h1 = op1(h1)
            s = h1
            states += [s]
        return s

class Cell4(nn.Module):

    def __init__(self, genotype, C):
        super(Cell4, self).__init__()
        print( C)

        cell4, indices4 = zip(*genotype.cell4)
        self._compile(C, cell4, indices4)

    def _compile(self, C, cell4, indices4):
        assert len(cell4) == len(indices4)
        self._steps = 1
        self.multiplier = 1

        self._ops = nn.ModuleList()
        for name, index in zip(cell4, indices4):
            op = OPS[name](C, 1, True)
            self._ops += [op]
        self._indices = indices4

    def forward(self, x):
        s0 = x
        states = [s0]
        for i in range(self._steps):
            h1 = states[self._indices[ i]]
            op1 = self._ops[i]
            h1 = op1(h1)
            s = h1
            states += [s]
        return s

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

class sun_model_trian(nn.Module):
    def __init__(self, genotype, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(sun_model_trian, self).__init__()
        self.pretrained = pretrained
        # self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.model.layer4[0].conv2.stride = (1,1)
        self.model.layer4[0].downsample[0].stride = (1,1)

        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveMaxPool2d(1)

        self._steps = 1

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.model.fc.in_features

            self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                #self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                #init.normal_(self.classifier.weight, std=0.001)
                self.classifier = ClassBlock(2048, self.num_classes, num_bottleneck=512)
        # init.constant_(self.feat_bn.weight, 1)
        # init.constant_(self.feat_bn.bias, 0)

        self.cell1 = Cell1(genotype, 256)
        self.cell2 = Cell2(genotype, 512)
        self.cell3 = Cell3(genotype, 1024)

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)

        x = self.cell1(x)
        x = self.model.layer2(x)

        x = self.cell2(x)
        x = self.model.layer3(x)

        x = self.cell3(x)
        x = self.model.layer4(x)

        x1 = self.gap1(x)
        x2 = self.gap2(x)
        x3 = x1+x2
        x3 = torch.squeeze(x3)

        #bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = self.classifier(x3)
            #bn_x = F.normalize(x3)
            return bn_x

        x3 = self.drop(x3)
        prob = self.classifier(x3)

        return x3, prob
