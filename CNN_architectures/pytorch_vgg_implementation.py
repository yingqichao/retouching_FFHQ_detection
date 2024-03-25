"""
A from scratch implementation of the VGG architecture.

Video explanation: https://youtu.be/ACmuBbuXn20
Got any questions leave a comment on youtube :)

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-05 Initial coding

"""

# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes:list=[1000]):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG19"])
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 512),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # nn.Linear(4096, 4096),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(4096, num_classes),
        )

        cls_head = nn.ModuleList([])
        for num_class in num_classes:
            cls_head.append(nn.Linear(512, num_class))
        setattr(self, f"cls_head", cls_head)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        # x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)

        ### todo: trans classification
        output_trans = []
        for idx, _ in enumerate(self.num_classes):
            output_trans.append(self.cls_head[idx](x))

        return [output_trans]

        # return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG_net(in_channels=3, num_classes=1000).to(device)
    print(model)
    ## N = 3 (Mini batch size)
    # x = torch.randn(3, 3, 224, 224).to(device)
    # print(models(x).shape)
