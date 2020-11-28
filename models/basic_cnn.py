# -*- coding: utf-8 -*-
__author__ = 'huangyf'

from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models


class CNN(nn.Module):
    def __init__(self, channels, output_size, with_bn=True):
        super(CNN, self).__init__()
        self.with_bn = with_bn
        self.features = self._make_layers(channels)
        self.classifier = nn.Linear(channels, output_size)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, channels):
        layers = []
        in_channels = 3
        for i in range(5):
            if i == 0:
                if self.with_bn:
                    layers += [('conv%d' % i, nn.Conv2d(in_channels, channels, 3, 2, 1)),
                               ('bn%d' % i, nn.BatchNorm2d(channels)),
                               ('relu%d' % i, nn.ReLU(inplace=True))]
                else:
                    layers += [('conv%d' % i, nn.Conv2d(in_channels, channels, 3, 2, 1)),
                               ('relu%d' % i, nn.ReLU(inplace=True))]
            else:
                if self.with_bn:
                    layers += [('conv%d' % i, nn.Conv2d(channels, channels, 3, 2, 1)),
                               ('bn%d' % i, nn.BatchNorm2d(channels)),
                               ('relu%d' % i, nn.ReLU(inplace=True))]
                else:
                    layers += [('conv%d' % i, nn.Conv2d(channels, channels, 3, 2, 1)),
                               ('relu%d' % i, nn.ReLU(inplace=True))]
        return nn.Sequential(OrderedDict(layers))
