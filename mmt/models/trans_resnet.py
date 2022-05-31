import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

######################################################################
class SEBlock(nn.Module):
  """ SE net Layer"""

  def __init__(self, channel, reduction=16):
    super(SEBlock, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(channel // reduction, channel, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)

class SpatialBlock(nn.Module):
  """ SpatialBlock net Layer"""

  def __init__(self, channel, input_dim, reduction=512):
    super(SpatialBlock, self).__init__()
    self.mean = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1)
    self.fc = nn.Sequential(
      nn.Linear(input_dim, reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(reduction, input_dim, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, h, w = x.size()
    y = self.mean(x)
    y = y.view(y.size(0), -1)
    y = self.fc(y)
    y = y.view(b, 1, h, w)
    return x * y.expand_as(x)

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

# |--Linear--|--bn--|--relu--|--Linear--|
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

# ft_net_50_1
class ft_net(nn.Module):
    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        self.avgpool_1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_4 = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = ClassBlock(2048, class_num, num_bottleneck=512)
        self.classifier_3 = ClassBlock(2048, class_num, num_bottleneck=512)
        self.classifier_4 = ClassBlock(2048, class_num, num_bottleneck=512)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        #x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)

        x1 = torch.squeeze(x)
        x2= self.classifier(x1)
        #
        return x1,x2