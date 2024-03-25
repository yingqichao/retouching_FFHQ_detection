import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

# model = resnet50()
from CNN_architectures.convnextv2_unet import ConvNeXtV2
from CNN_architectures.pytorch_resnet import block


class ConvUnet_plugin(ConvNeXtV2):
    def __init__(self, opt, img_size=512, num_classes:list=[1000,1000], layers=[3,4,6,3], embed_dims=[96,192,384,768],
                 drop_tokens=64, downsample_times=[2, 1, 1, 1], depth_transformer=4,):
        # model = resnet50() # Bottleneck, [3, 4, 6, 3], weights: None, progress: True
        # block = torchvision.models.resnet.Bottleneck
        super(ConvUnet_plugin, self).__init__()
        self.num_classes = num_classes
        self.unified_dim = embed_dims[-1]
        self.opt = opt
        ## todo: feature projection: BCHW->BLC
        for i in range(4):
            # ## todo: feature calibration
            if "skip_calibrate" in self.opt:
                print("please note that we are skipping the calibration module!")
            else:
                calibrate = nn.Sequential(
                    *[block(in_channels=embed_dims[i], intermediate_channels=embed_dims[i], expansion=1) for j in range(1)],
                )
                setattr(self, f"calibrate_CNN_{i}", calibrate)

            downsample = nn.Sequential(
                nn.AdaptiveAvgPool2d((2 ** (3 - i), 2 ** (3 - i))),
                nn.Flatten(),
                nn.Linear(embed_dims[i] * (2 ** (3 - i)) * (2 ** (3 - i)), self.unified_dim)
            )

            setattr(self, f"plogin_mlp_{i}", downsample)

        ## todo: positional encoding
        hier_MLP = nn.Sequential(
            nn.Linear(4 * self.unified_dim, self.unified_dim),
            nn.SiLU(),
        )
        setattr(self, f"hier_MLP", hier_MLP)
        ##### Classifier head
        # self.trans_norm = nn.LayerNorm(embed_dim)
        # self.cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        cls_head = nn.ModuleList([])
        for num_class in self.num_classes:
            cls_head.append(nn.Linear(self.unified_dim, num_class))
        setattr(self, f"cls_head", cls_head)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)
        # CNN_cls_head = nn.ModuleList([])
        # for num_class in num_classes:
        #     CNN_cls_head.append(nn.Linear(self.unified_dim, num_class))
        # setattr(self, f"CNN_cls_head", CNN_cls_head)
        # trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'token_type_embeddings',
                'cls_token'}


    def forward(self,x):
        B = x.shape[0]
        x, x_hier = self.Encoder(x)
        ## todo: reproduce the original image
        x_decoded = self.Decoder(x, x_hier)
        x_decoded = self.out_conv(x_decoded)

        hierarchical_tokens = []
        for i in range(4):
            x_proj = x_hier[i]
            if "skip_calibrate" not in self.opt:
                calibrate = getattr(self, f"calibrate_CNN_{i}")
                x_proj = calibrate(x_proj)
            plogin_mlp = getattr(self, f"plogin_mlp_{i}")
            x_proj = plogin_mlp(x_proj)
            hierarchical_tokens.append(x_proj)

        ### todo: transformer replaced by MLP
        attended_tokens = self.hier_MLP(torch.cat(hierarchical_tokens, dim=1))

        ### todo: trans classification
        output_trans = []
        for idx, _ in enumerate(self.num_classes):
            output_trans.append(self.cls_head[idx](attended_tokens))

        return x_decoded, [output_trans]

    def forward_classification(self,x):
        B = x.shape[0]
        x, x_hier = self.Encoder(x)

        hierarchical_tokens = []
        for i in range(4):
            x_proj = x_hier[i]
            if "skip_calibrate" not in self.opt:
                calibrate = getattr(self, f"calibrate_CNN_{i}")
                x_proj = calibrate(x_proj)
            plogin_mlp = getattr(self, f"plogin_mlp_{i}")
            x_proj = plogin_mlp(x_proj)
            hierarchical_tokens.append(x_proj)

        ### todo: transformer replaced by MLP
        attended_tokens = self.hier_MLP(torch.cat(hierarchical_tokens, dim=1))

        ### todo: trans classification
        output_trans = []
        for idx, _ in enumerate(self.num_classes):
            output_trans.append(self.cls_head[idx](attended_tokens))

        return [output_trans]


if __name__ == '__main__':
    # inference(input=2)
    model = ConvUnet_plugin(opt={'skip_calibrate':True})
    output = model(torch.rand((1, 3, 512, 512)))
    print(output)
    # model = custom_viT(img_size=224,num_classes=[1000,1000])
    # model(torch.rand((1,3,224,224)))