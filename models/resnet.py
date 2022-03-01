import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from models.blocks import *
import MinkowskiEngine as ME

class SparseResNet(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)
    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.expansion = 4
        self.layers = []
        self.D = D
        self.inplanes = self.INIT_DIM

        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels,
                self.inplanes,
                kernel_size=5,
                stride=2,
                dimension=D,
            ),
            ME.MinkowskiBatchNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiSumPooling(kernel_size=in_channels, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D,
            ),
            ME.MinkowskiBatchNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample=None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )

        layers = []

        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )

        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        h = self.conv5(x)

        return h

class SparseResNet14(SparseResNet):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)

class SparseResNet18(SparseResNet):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)

class SparseResNet34(SparseResNet):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)

class SparseResNet50(SparseResNet):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)

class SparseResNet101(SparseResNet):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)
