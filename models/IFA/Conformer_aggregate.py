import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class Block(nn.Module):
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

from einops import rearrange
class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape, num_tokens=50):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
                            torch.nn.Linear(input_shape, input_shape),
                            nn.Dropout(0.1),
                            nn.SiLU(),
                            #SimpleGate(dim=2),
                            torch.nn.Linear(input_shape, num_tokens),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs)
        scores = rearrange(scores, 'b l s -> b s l')
        # scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        # scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # scores = self.attention_layer(inputs)
        # outputs = scores*inputs
        return outputs



from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D, Summer
import models.IFA.convnextv2_official as Convnext
import models.IFA.shunted_transformer as ShuntedTransformer
import models.IFA.custom_block as custom_block
from models.IFA.GroupViT import GroupingBlock
class Conformer_hierarchical(nn.Module):

    def __init__(self, in_chans=3, num_classes=[6,3], num_heads=[8,8,8,8], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 img_size=512, depths_cnn=[2,6,2,2], embed_dims=[64, 128, 256, 512], sr_ratios=[4, 2, 1],
                 norm_layer=nn.LayerNorm, mlp_ratios=[4, 4, 4], num_conv=2, downsample_times=[3, 2, 1, 1],
                 drop_tokens_per_layer=64, depth_transformer=4,
                 ):
        super().__init__()


        self.downsample_times = downsample_times
        self.num_stages = len(depths_cnn)
        self.num_classes = num_classes
        self.num_cls_tokens = len(num_classes)
        self.unified_dim = embed_dims[-1]


        ## todo: CNN feature extraction
        for i in range(self.num_stages):
            ## todo: downsample and backbone
            for j in range(self.downsample_times[i]):
                downsample = ShuntedTransformer.OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** i),
                                                         patch_size=7 if i==0 and j==0 else 3,
                                                         stride=2 if i==0 and j==0 else 2,
                                                         in_chans=(3 if i==0 else embed_dims[i-1]) if j==0 else embed_dims[i],
                                                         embed_dim=embed_dims[i])

                stage = nn.Sequential(
                    *[Convnext.Block(dim=embed_dims[i], drop_path=0.1) for j in range(depths_cnn[i])]
                )
                setattr(self, f"downsample_CNN_{i}_{j}", downsample)
                setattr(self, f"stem_CNN_{i}_{j}", stage)


            # ## todo: feature calibration
            calibrate = nn.Sequential(
                *[Convnext.Block(dim=embed_dims[i], drop_path=0.1) for j in range(2)],
            )
            setattr(self, f"calibrate_CNN_{i}", calibrate)

            ## todo: feature projection: BCHW->BLC
            downsample = nn.Sequential(
                nn.AdaptiveAvgPool2d((2 ** (3 - i), 2 ** (3 - i))),
                nn.Flatten(),
                nn.Linear(embed_dims[i] * (2 ** (3 - i)) * (2 ** (3 - i)), self.unified_dim)
            )


            setattr(self, f"project_CNN_{i}", downsample)

        ## todo: final layer attention
        # final_layer_attention = TokenAttention(self.unified_dim, len(self.num_classes))
        #
        # setattr(self, f"final_layer_attention", final_layer_attention)
        # ## todo: Shunted transformer for global comparison
        # block = nn.ModuleList(
        #     [
        #         custom_block.Block(
        #             dim=self.unified_dim,
        #             num_heads=16,
        #             n_experts=1,
        #             drop=0.1,
        #             drop_path=0.1,
        #             attn_drop=0.1,
        #         )
        #         for i in range(depth_transformer)
        #     ]
        # )
        #
        # setattr(self, f"trans", block)

        ## todo: positional encoding
        hier_MLP = nn.Sequential(
            nn.Linear(4*self.unified_dim,self.unified_dim),
            nn.SiLU(),
        )
        setattr(self, f"hier_MLP", hier_MLP)

        trans_cls_head = nn.ModuleList([])
        for num_class in num_classes:
            trans_cls_head.append(nn.Linear(self.unified_dim, num_class))
        setattr(self, f"trans_cls_head", trans_cls_head)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)
        CNN_cls_head = nn.ModuleList([])
        for num_class in num_classes:
            CNN_cls_head.append(nn.Linear(self.unified_dim, num_class))
        setattr(self, f"CNN_cls_head", CNN_cls_head)
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

    def forward_features(self, x):
        # B = x.shape[0]
        # feats_pre = []
        # feats_transformer = []
        # x, _, _ = self.stem_embed(x, flatten=False)
        # x,_,_ = self.stem(x, flatten=False)
        B = x.shape[0]
        tokens = []


        for i in range(self.num_stages):
            ## every step: patch embed, Block, norm
            for j in range(self.downsample_times[i]):
                down = getattr(self, f"downsample_CNN_{i}_{j}")
                stage = getattr(self, f"stem_CNN_{i}_{j}")

                x, _, _ = down(x, flatten=False)
                x = stage(x)

            # x_proj = rearrange(x, 'b c h w -> b (h w) c')

            calibrate = getattr(self, f"calibrate_CNN_{i}")
            project = getattr(self, f"project_CNN_{i}")

            x_proj = calibrate(x)
            x_proj = project(x_proj)

            tokens.append(x_proj)


        return tokens, x #, feats_transformer

    def forward(self, x):
        B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        # x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        ### todo: CNN
        hierarchical_tokens, CNN_final = self.forward_features(x)

        ### todo: transformer replaced by MLP
        attended_tokens = self.hier_MLP(torch.cat(hierarchical_tokens,dim=1))

        ### todo: trans classification
        output_trans = []
        for idx, _ in enumerate(self.num_classes):
            output_trans.append(self.trans_cls_head[idx](attended_tokens))
        # output.append(output_trans)

        return [output_trans]

if __name__ == '__main__':
    # model = Conformer_hierarchical(
    #     patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 24, 2], sr_ratios=[8, 4, 2, 1], num_conv=2, num_classes=[6,6]
    # )
    # x = torch.randn(1, 3, 512, 512)
    # output_cnn, output_trans = model(x)
    # print(output_cnn[0].shape)
    # print(output_trans[0].shape)
    attention = TokenAttention(768)
    input = torch.ones((2,197,768))
    text_atn_feature, _ = attention(input)
    print(text_atn_feature.shape)