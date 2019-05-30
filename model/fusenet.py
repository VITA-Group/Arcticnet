import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .resnet import resnet50
from .resnet import Bottleneck


def FuseNet(name, num_classes):
    if name == "lateFuse":
        return FuseNet_late(num_classes)
    elif name == "midFuse_layer3":
        return FuseNet_midLayer3(num_classes)
    elif name == "midFuse_layer2":
        return FuseNet_midLayer2(num_classes)
    elif name == "earlyFuse":
        return FuseNet_early(num_classes)
    else:
        raise Exception("expect FUSEMODE to be one of lateFuse, midFuse_layer2, midFuse_layer3 and earlyFuse")


class FuseNet_late(nn.Module):
    def __init__(self, num_classes):
        super(FuseNet_late, self).__init__()
        self.resnet1 = resnet50(num_classes=num_classes, pretrained=True)
        self.resnet2 = resnet50(num_classes=num_classes, pretrained=True)
        self.fc = nn.Linear(512 * 8, num_classes)

    def forward(self, image1, image2):

        with torch.no_grad():
            x1 = self.resnet1.conv1(image1)
            x1 = self.resnet1.bn1(x1)
            x1 = self.resnet1.relu(x1)
            x1 = self.resnet1.maxpool(x1)

            x1 = self.resnet1.layer1(x1)
            x1 = self.resnet1.layer2(x1)
            x1 = self.resnet1.layer3(x1)
            x1 = self.resnet1.layer4(x1)

            x1 = F.avg_pool2d(x1, x1.size()[2:])
            x1 = x1.view(x1.size(0), -1)

            x2 = self.resnet2.conv1(image2)
            x2 = self.resnet2.bn1(x2)
            x2 = self.resnet2.relu(x2)
            x2 = self.resnet2.maxpool(x2)

            x2 = self.resnet2.layer1(x2)
            x2 = self.resnet2.layer2(x2)
            x2 = self.resnet2.layer3(x2)
            x2 = self.resnet2.layer4(x2)

            x2 = F.avg_pool2d(x2, x2.size()[2:])
            x2 = x2.view(x2.size(0), -1)
        x = self.fc(torch.cat((x1, x2), 1))

        return x


class FuseNet_early(nn.Module):
    def __init__(self, num_classes):
        super(FuseNet_early, self).__init__()
        self.resnet = resnet50(num_classes=num_classes, pretrained=True, in_channel=6)

    def forward(self, image1, image2):

        x = self.resnet(torch.cat([image1, image2], 1))

        return x


class FuseNet_midLayer3(nn.Module):
    def __init__(self, num_classes):
        super(FuseNet_midLayer3, self).__init__()
        self.resnet1 = resnet50(num_classes=num_classes, pretrained=True)
        self.resnet2 = resnet50(num_classes=num_classes, pretrained=True)
        self.resnet1.inplanes = 2048
        self.layer4 = self.resnet1._make_layer(Bottleneck, 512, 3, stride=2)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, image1, image2):

        with torch.no_grad():
            x1 = self.resnet1.conv1(image1)
            x1 = self.resnet1.bn1(x1)
            x1 = self.resnet1.relu(x1)
            x1 = self.resnet1.maxpool(x1)

            x1 = self.resnet1.layer1(x1)
            x1 = self.resnet1.layer2(x1)
            x1 = self.resnet1.layer3(x1)
            # x1 = self.resnet1.layer4(x1)

            x2 = self.resnet2.conv1(image2)
            x2 = self.resnet2.bn1(x2)
            x2 = self.resnet2.relu(x2)
            x2 = self.resnet2.maxpool(x2)

            x2 = self.resnet2.layer1(x2)
            x2 = self.resnet2.layer2(x2)
            x2 = self.resnet2.layer3(x2)
            # x2 = self.resnet2.layer4(x2)

        x = self.layer4(torch.cat([x1, x2], 1))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class FuseNet_midLayer2(nn.Module):
    def __init__(self, num_classes):
        super(FuseNet_midLayer2, self).__init__()
        self.resnet1 = resnet50(num_classes=num_classes, pretrained=True)
        self.resnet2 = resnet50(num_classes=num_classes, pretrained=True)
        self.resnet1.inplanes = 1024
        self.layer3 = self.resnet1._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self.resnet1._make_layer(Bottleneck, 512, 3, stride=2)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, image1, image2):

        with torch.no_grad():
            x1 = self.resnet1.conv1(image1)
            x1 = self.resnet1.bn1(x1)
            x1 = self.resnet1.relu(x1)
            x1 = self.resnet1.maxpool(x1)

            x1 = self.resnet1.layer1(x1)
            x1 = self.resnet1.layer2(x1)
            # x1 = self.resnet1.layer3(x1)
            # x1 = self.resnet1.layer4(x1)

            x2 = self.resnet2.conv1(image2)
            x2 = self.resnet2.bn1(x2)
            x2 = self.resnet2.relu(x2)
            x2 = self.resnet2.maxpool(x2)

            x2 = self.resnet2.layer1(x2)
            x2 = self.resnet2.layer2(x2)
            # x2 = self.resnet2.layer3(x2)
            # x2 = self.resnet2.layer4(x2)

        x = self.layer3(torch.cat([x1, x2], 1))
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

