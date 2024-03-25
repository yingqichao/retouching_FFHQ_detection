import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from CNN_architectures.custom_densenet import custom_densenet
from models.IFA.Conformer_token_attention import Mlp, TokenAttention
from timm.models.layers import DropPath, trunc_normal_
import models.IFA.custom_block as custom_block
import torchvision
# model = resnet50()
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D, Summer
from einops import rearrange
from models.IFA.GroupViT import GroupingBlock
from CNN_architectures.pytorch_resnet import ResNet, block

class densenet_plugin(custom_densenet):
    def __init__(self, opt, img_size=512, num_classes:list=[1000,1000], embed_dims=[256,512,1024,1024], num_heads=[8,8,8,8],
                 drop_tokens=64, downsample_times=[2, 1, 1, 1], depth_transformer=2,):
        # model = resnet50() # Bottleneck, [3, 4, 6, 3], weights: None, progress: True
        # block = torchvision.models.resnet.Bottleneck
        super(densenet_plugin, self).__init__(num_classes=num_classes)
        self.opt = opt
        self.unified_dim = embed_dims[-1]
        self.num_classes = num_classes
        self.drop_tokens_per_layer = [1. / drop_tokens,
                                 1. / (drop_tokens / 4),
                                 1. / (drop_tokens / 16),
                                 1.0
                                 ]  # default: [1/64,1/16,1/4,1.0]

        self.downsample_times = downsample_times
        self.num_tokens_per_layer = [
            (img_size // (2 ** (downsample_times[0]))) ** 2,
            (img_size // (2 ** (downsample_times[0] + downsample_times[1]))) ** 2,
            (img_size // (2 ** (downsample_times[0] + downsample_times[1] + downsample_times[2]))) ** 2,
            (img_size // (2 ** (
                        downsample_times[0] + downsample_times[1] + downsample_times[2] + downsample_times[3]))) ** 2,
        ]

        self.num_tokens_after_grouping = [int(self.num_tokens_per_layer[i] * self.drop_tokens_per_layer[i])
                                          for i in range(len(self.num_tokens_per_layer))]

        ## todo: plugin - token attention
        for i in range(4):
            # ## todo: feature calibration
            if "skip_calibrate" in self.opt:
                print("please note that we are skipping the calibration module!")
            else:
                calibrate = nn.Sequential(
                    *[block(in_channels=embed_dims[i], intermediate_channels=embed_dims[i], expansion=1) for j in
                      range(1)],
                )
                setattr(self, f"calibrate_CNN_{i}", calibrate)

            mlp_proj = nn.Identity() if embed_dims[i] == self.unified_dim else \
                Mlp(in_features=embed_dims[i], out_features=self.unified_dim)
            group_embeddings = nn.Embedding(self.num_tokens_after_grouping[i], embed_dims[i])
            position_group_tokens = torch.arange(start=0, end=self.num_tokens_after_grouping[i]).cuda()
            setattr(self, f"group_embeddings_{i}", group_embeddings)
            setattr(self, f"position_group_tokens_{i}", position_group_tokens)
            token_attention = GroupingBlock(
                dim=embed_dims[i],
                out_dim=self.unified_dim,
                num_heads=num_heads[i],
                num_group_token=self.num_tokens_after_grouping[i],
                num_output_group=self.num_tokens_after_grouping[i],
            )

            # token_attention = nn.Identity() if self.drop_tokens_per_layer[i] == 1.0 else \
            #         TokenAttention(embed_dims[i], self.num_tokens_after_grouping[i])
            setattr(self, f"plugin_token_attention_{i}", token_attention)
            setattr(self, f"plogin_mlp_{i}", mlp_proj)

        ## todo: final layer attention
        final_layer_attention = TokenAttention(self.unified_dim, len(self.num_classes))

        setattr(self, f"final_layer_attention", final_layer_attention)
        ## todo: Shunted transformer for global comparison
        trans_block = nn.ModuleList(
            [
                custom_block.Block(
                    dim=self.unified_dim,
                    num_heads=16,
                    n_experts=1,
                    drop=0.1,
                    drop_path=0.1,
                    attn_drop=0.1,
                )
                for i in range(depth_transformer)
            ]
        )

        setattr(self, f"trans", trans_block)

        ## todo: positional encoding
        self.num_cls_tokens = len(num_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_cls_tokens, self.unified_dim))
        self.position_1d = Summer(PositionalEncoding1D(self.unified_dim))
        self.token_type_embeddings = nn.Embedding(4, self.unified_dim)
        self.token_type_list = torch.tensor(
            # [0] * self.num_cls_tokens +
            [0] * int(self.num_tokens_per_layer[0] * self.drop_tokens_per_layer[0]) +
            [1] * int(self.num_tokens_per_layer[1] * self.drop_tokens_per_layer[1]) +
            [2] * int(self.num_tokens_per_layer[2] * self.drop_tokens_per_layer[2]) +
            [3] * int(self.num_tokens_per_layer[3] * self.drop_tokens_per_layer[3])
        ).cuda()

        ##### Classifier head
        # self.trans_norm = nn.LayerNorm(embed_dim)
        # self.cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        cls_head = nn.ModuleList([])
        for num_class in num_classes:
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

    def forward(self, x):
        B = x.shape[0]
        x, x_hier = self.forward_feature(x)
        hierarchical_tokens = []
        for i in range(4):
            x_proj = x_hier[i]
            if "skip_calibrate" not in self.opt:
                calibrate = getattr(self, f"calibrate_CNN_{i}")
                x_proj = calibrate(x_proj)

            x_proj = rearrange(x_proj, 'b c h w -> b (h w) c')
            position_group_tokens = getattr(self, f"position_group_tokens_{i}")
            group_embeddings = getattr(self, f"group_embeddings_{i}")
            plugin_token_attention = getattr(self, f"plugin_token_attention_{i}")
            # plogin_mlp = getattr(self, f"plogin_mlp_{i}")
            positions = position_group_tokens.repeat(B, 1)
            group_token = group_embeddings(positions)
            x_proj, attention_map = plugin_token_attention(x_proj, group_token)
            # x_proj = plogin_mlp(x_proj)
            hierarchical_tokens.append(x_proj)

        ### todo: positional embeddings and modality embeddings
        for tokens in hierarchical_tokens:
            ## todo: warning, might not be necessary because CNN feats can carry positional information
            pos_emb = self.position_1d(tokens)
            tokens += pos_emb

        tokens = torch.cat(hierarchical_tokens, dim=1)

        # cls_token = self.cls_token.expand(B, -1, -1)
        token_type_list = self.token_type_list.repeat(B, 1)
        token_encoding = self.token_type_embeddings(token_type_list)

        # tokens = torch.cat([cls_token, tokens], dim=1)
        tokens += token_encoding

        ### todo: transformer
        for idx, blk in enumerate(self.trans):
            tokens = blk(tokens, indexes_list=token_type_list.long())

        ### todo: final layer attention: [B, classes, dim]
        attended_tokens = self.final_layer_attention(tokens)

        ### todo: trans classification
        output_trans = []
        for idx, _ in enumerate(self.num_classes):
            output_trans.append(self.cls_head[idx](attended_tokens[:, idx]))

        return [output_trans]


if __name__ == '__main__':
    # inference(input=2)
    model = densenet_plugin(opt={'skip_calibrate':True})
    output = model(torch.rand((1, 3, 512, 512)))
    print(output)
    # model = custom_viT(img_size=224,num_classes=[1000,1000])
    # model(torch.rand((1,3,224,224)))