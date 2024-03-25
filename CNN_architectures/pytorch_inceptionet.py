"""
An implementation of GoogLeNet / InceptionNet from scratch.

Video explanation: https://youtu.be/uQc4Fs7yx5I
Got any questions leave a comment on youtube :)

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-07 Initial coding

"""

# Imports
import torch
from torch import nn
# import antialiased_cnns

import numpy as np
class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=False, num_classes:list=[1000], use_SRM=False, antialias=True):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
        self.use_SRM = use_SRM
        self.num_classes = num_classes
        # Write in_channels, etc, all explicit in self.conv1, rest will write to
        # make everything as compact as possible, kernel_size=3 instead of (3,3)
        image_channels = 3
        if self.use_SRM:
            print("We are using SRM in Inception")
            ## bayar conv
            self.BayarConv2D = nn.Conv2d(3, self.use_SRM, 5, 1, padding=2, bias=False)
            self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
            self.bayar_mask[2, 2] = 0
            self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
            self.bayar_final[2, 2] = -1

            # ## srm conv
            # self.SRMConv2D = nn.Conv2d(3, 9, 5, 1, padding=2, bias=False)
            # self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
            #
            # ##SRM filters (fixed)
            # for param in self.SRMConv2D.parameters():
            #     param.requires_grad = False

            # self.relu = nn.SiLU() #nn.ELU(inplace=True)
            # image_channels = 12


        self.conv1 = conv_block(
            in_channels=image_channels*2 if self.use_SRM else image_channels,
            out_channels=64,
            xbn=False,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )

        print("We are using antialias in Inception")
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool1 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #     antialiased_cnns.BlurPool(64,stride=2),
        # )

        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool2 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #     antialiased_cnns.BlurPool(192, stride=2),
        # )

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        # self.maxpool3 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #     antialiased_cnns.BlurPool(480, stride=2),
        # )

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool4 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #     antialiased_cnns.BlurPool(832, stride=2),
        # )

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4)
        # self.fc1 = nn.Linear(1024, num_classes)
        self.fc1 = nn.ModuleList([])
        for num_class in num_classes:
            self.fc1.append(nn.Sequential(
                    nn.Linear(1024, num_class),
                )
            )

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward_feature(self, x):
        x_hier = []
        if self.use_SRM:
            self.BayarConv2D.weight.data *= self.bayar_mask
            self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1),-1)
            self.BayarConv2D.weight.data += self.bayar_final
            conv_bayar = self.BayarConv2D(x)
            # conv_srm = self.SRMConv2D(x)

            x = conv_bayar #torch.cat((conv_srm, conv_bayar), dim=1)
            # x = self.relu(x)

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x_hier.append(x)
        # x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x_hier.append(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary Softmax classifier 1
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x_hier.append(x)

        # Auxiliary Softmax classifier 2
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x_hier.append(x)
        return x, x_hier

    def forward(self, x):
        x, x_hier = self.forward_feature(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        # x = self.fc1(x)

        cls = []
        for idx, _ in enumerate(self.num_classes):
            cls.append(self.fc1[idx](x.view(x.shape[0], -1)))
        return [cls]

        # if self.aux_logits and self.training:
        #     return aux1, aux2, x
        # else:
        #     return x


class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, xbn=False, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, xbn=False, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim=dim

    def forward(self,x):
        x1,x2 = x.chunk(2,dim=self.dim)
        return x1*x2


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, xbn=False, **kwargs):
        super(conv_block, self).__init__()
        # print(f"Using group norm: {xbn}")
        self.relu = nn.SiLU() #SimpleGate() #nn.ReLU() nn.SiLU(),
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels) if not xbn else nn.GroupNorm(num_channels=out_channels,num_groups=16)
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=1,padding=0,stride=1,
        #               groups=1,bias=True,
        #     ),
        # )

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # x = x*self.sca(x)
        x = self.batchnorm(x)
        return x
        # return self.relu(self.batchnorm(self.conv(x)))


if __name__ == "__main__":
    # N = 3 (Mini batch size)

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    with torch.no_grad():
        from thop import profile

        # from lama_models.HWMNet import HWMNet

        nin, nout = 3, 1
        X = torch.randn(1, nin, 512, 512).cuda()
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.memory_reserved())
        # model = SKFF(in_channels=16)
        # X = [torch.randn(1, 16, 64, 64), torch.randn(1, 16, 64, 64), torch.randn(1, 16, 64, 64)]
        # print(X.shape)

        ## todo: network definition
        model = GoogLeNet(num_classes=[1000, 1000, 1000, 1000]).cuda()
        # model = HWMNet(in_chn=1, out_chn=1, wf=32, depth=4, subtask=0, style_control=False, use_dwt=False).cuda()
        # Y = model(X)
        # print(Y.shape)

        flops, params = profile(model, (X,))
        print(flops)
        print(params)
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())
