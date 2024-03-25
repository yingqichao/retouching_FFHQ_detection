import torch
import torch.nn as nn
from torchvision.models.densenet import DenseNet
import torch.nn.functional as F

class custom_densenet(DenseNet):
    def __init__(self, num_classes:list=[1000]):
        super(custom_densenet, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.classifier.weight.shape[1]
        self.head = nn.ModuleList([])
        for num_class in num_classes:
            self.head.append(nn.Linear(self.num_features, num_class))

    def forward_feature(self, x):
        x_hier = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if "DenseBlock" in layer._get_name():
                x_hier.append(x)
        return x, x_hier  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        cls = []
        for idx, _ in enumerate(self.num_classes):
            cls.append(self.head[idx](out))
        return [cls]

    def get_last_layer(self):
        return self.features[-1]


if __name__ == '__main__':
    model = custom_densenet(num_classes=[1000])
    model.forward_features(torch.rand((1,3,224,224)))