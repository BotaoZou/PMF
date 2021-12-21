from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def ConvTransposeBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, output_padding=output_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def DownSample(kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
    )


class InceptionV2(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce,
                 out_channels3, out_channels4):
        super().__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class MVLNet_stage1(nn.Module):
    def __init__(self, pcd_channels=3, nclasses=7):
        super().__init__()
        self.pcd_channels = pcd_channels
        self.nclasses = nclasses
        self.trunk1 = ConvBNReLU(pcd_channels, 64, 3, 1, 1)
        self.trunk2 = ConvBNReLU(64, 64, 3, 1, 1)
        self.trunk3 = ConvBNReLU(64, 128, 3, 1, 1)

        # inception模块的输出channel怎么确定？
        self.block1_inception = InceptionV2(128, 16, 16, 16, 16, 16, 16)
        self.block2_inception = InceptionV2(64, 16, 16, 16, 16, 16, 16)
        self.block3_inception = InceptionV2(64, 32, 32, 32, 32, 32, 32)

        self.deconv_1a = ConvTransposeBNReLU(128, 256, 3, 2, 1, 1)
        self.deconv_2a = ConvTransposeBNReLU(256, 128, 3, 2, 1, 1)
        self.deconv_3a = ConvTransposeBNReLU(128, 64, 3, 2, 1, 1)

        self.downsample = DownSample(3, 2, 1)

    def forward(self, x, img_feature=[]):
        # downsamping
        output = self.trunk1(x)
        output = self.trunk2(output)
        output = self.trunk3(output)
        output = self.downsample(output)

        output = self.block1_inception(output)
        block1 = self.block1_inception(output)

        output = self.block2_inception(block1)
        output = self.block2_inception(output)
        block2 = self.downsample(output)

        output = self.block3_inception(block2)
        output = self.block3_inception(output)
        output = self.block3_inception(output)
        block3 = self.downsample(output)

        # unsampling
        output = self.deconv_1a(block3)
        output = torch.cat([block2, output], dim=1)
        output = ConvBNReLU(320, 256, 1, 1, 0)(output)
        output = ConvBNReLU(256, 256, 3, 1, 1)(output)
        output = self.deconv_2a(output)
        output = torch.cat([block1, output], dim=1)
        output = ConvBNReLU(192, 128, 1, 1)(output)
        output = ConvBNReLU(128, 128, 3, 1, 1)(output)
        output = self.deconv_3a(output)
        output = ConvBNReLU(64, 64, 1, 1)(output)
        output = ConvBNReLU(64, 64, 3, 1, 1)(output)

        # classhead
        output = ConvBNReLU(64, 64, 3, 1, 1)(output)
        output = ConvBNReLU(64, self.nclasses, 1, 1)(output)
        return output


class MVLNet_stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_base = nn.Sequential(
            ConvBNReLU(7, 16, 3, 1, 1),
            ConvBNReLU(16, 16, 3, 1, 1),
            ConvBNReLU(16, 16, 3, 1, 1),

            ConvBNReLU(32, 32, 3, 1, 1)
        )

        self.block1 = nn.Sequential(
            ConvBNReLU(3, 16, 3, 1, 1),
            ConvBNReLU(16, 16, 3, 1, 1),
            ConvBNReLU(16, 16, 3, 1, 1),
            ConvBNReLU(32, 32, 3, 1, 1)
        )

        self.block1.train(True)

    def forward(self, x, img_feature=[]):
        print("asd")
