import torch
import torch.nn as nn
import math

from .shakedrop import ShakeDrop

def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def BasicBlock(nn.Module):
    
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_shakedrop=1.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorn2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorn2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorn2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        out = self.shake_drop(out)
        
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = out.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                featuremap_size[1]).fillna_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut
        
        return out
