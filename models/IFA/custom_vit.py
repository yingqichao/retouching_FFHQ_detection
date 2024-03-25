import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torchvision.models.resnet import resnet50, ResNet

import torchvision
model = resnet50()
class custom_resnet50(ResNet):
    def __init__(self, img_size=224,num_classes:list=[1000,1000], layers=[3,4,6,3]):
        # model = resnet50() # Bottleneck, [3, 4, 6, 3], weights: None, progress: True
        block = torchvision.models.resnet.Bottleneck
        super(custom_resnet50, self).__init__(block=torchvision.models.resnet.Bottleneck,
                                              layers=layers)
        self.num_classes = num_classes
        self.out = nn.Linear(block.expansion*512,100)
        # self.num_features = self.fc.weight.shape[1]
        # self.head = nn.ModuleList([])
        # for num_class in num_classes:
        #     self.head.append(nn.Linear(self.num_features, num_class))

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.out(x)

        return x

        # return x

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     cls = []
    #     for idx, _ in enumerate(self.num_classes):
    #         cls.append(self.head[idx](x[:,0]))
    #     return [cls]



class custom_viT(VisionTransformer):
    def __init__(self, img_size=224,num_classes:list=[1000]):
        super(custom_viT, self).__init__(img_size)
        self.num_classes = num_classes
        self.head = nn.ModuleList([])
        for num_class in num_classes:
            self.head.append(nn.Linear(self.num_features, num_class))


    def forward(self, x):
        x = self.forward_features(x)
        cls = []
        for idx, _ in enumerate(self.num_classes):
            cls.append(self.head[idx](x[:,0]))
        return [cls]

# from beartype import beartype
#
# @beartype
def inference(input: list=[10,10]):
    print(input)

if __name__ == '__main__':
    # inference(input=2)
    model = custom_resnet50()
    output = model(torch.rand((1, 3, 224, 224)))
    print(output)
    # model = custom_viT(img_size=224,num_classes=[1000,1000])
    # model(torch.rand((1,3,224,224)))