import torch
import torch.nn as nn
from mmt.NAS.CBAM import ChannelAttention, SpatialAttention
from torch.nn import functional as F
from mmt.NAS.DANet import PositionAttentionModule, ChannelAttentionModule
import numpy as np

OPS = {
  'SEblock': lambda C, stride, affine: SEBlock(C),
  'SPblock': lambda C, stride, affine: SpatialBlock(C),
  'CBAMblock': lambda C, stride, affine: CBAMBlock(C),
  'Selfblock': lambda C, stride, affine:Self_Attn(C),
}


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
    y = self.fc(y)
    y = y.view(b, c, 1, 1)
    x1 = x * y.expand_as(x)
    return x1


def fc_(x, input_dim, reduction):
    fc = nn.Sequential(
    nn.Linear(input_dim, reduction, bias=False).cuda(),
    nn.ReLU(inplace=True).cuda(),
    nn.Linear(reduction, input_dim, bias=False).cuda(),
    nn.Sigmoid().cuda()
    )
    return fc(x)

class SpatialBlock(nn.Module):
  """ SpatialBlock net Layer"""

  def __init__(self, channel, reduction=512):
    super(SpatialBlock, self).__init__()
    self.mean = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1)
    self.reduction = reduction

  def forward(self, x):
    b, c, h, w = x.size()
    y = self.mean(x)
    y = y.view(y.size(0), -1).cuda()
    b_y, c_y =y.size()
    y = fc_(y, c_y, self.reduction)
    y = y.view(b, 1, h, w)
    x1 = x * y.expand_as(x)
    return x1


class CBAMBlock(nn.Module):

  def __init__(self, channel=512, reduction=16, kernel_size=7):
    super().__init__()
    self.ca = ChannelAttention(channel=channel, reduction=reduction)
    self.sa = SpatialAttention(kernel_size=kernel_size)

  def forward(self, x):
    b, c, _, _ = x.size()
    residual = x
    out = x * self.ca(x)
    out = out * self.sa(out)
    return out + residual

class DAModule(nn.Module):

  def __init__(self, d_model, kernel_size=3, H=7, W=7):
    super().__init__()
    self.position_attention_module = PositionAttentionModule(d_model, kernel_size=3, H=7, W=7)
    self.channel_attention_module = ChannelAttentionModule(d_model, kernel_size=3, H=7, W=7)

  def forward(self, x):
    bs, c, h, w = x.size()
    p_out = self.position_attention_module(x)
    c_out = self.channel_attention_module(x)
    p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)
    c_out = c_out.view(bs, c, h, w)
    return p_out + c_out


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        #self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out





