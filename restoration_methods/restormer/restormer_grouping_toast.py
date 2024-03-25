## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from models.vit_topdown.vit_top_down import Block
from models.IFA.GroupViT import GroupingBlock_Prompt


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, td=None):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        if td is not None:
            qkv_td = self.qkv_dwconv(self.qkv(td))
            q_td, k_td, v_td = qkv_td.chunk(3, dim=1)
            q_td = rearrange(q_td, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k_td = rearrange(k_td, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v_td = rearrange(v_td, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            q = q + q_td
            k = k + k_td
            v = v + v_td

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
from models.IFA.convnext_official import Block as ConvNextBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, td=None):
        x = x + self.attn(self.norm1(x), td=td)
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
from functools import partial
from models.vit_topdown.vit_top_down import Decode_Block
import numpy as np
class Restormer(nn.Module):
    def __init__(self,
                 img_size=None,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 drop_tokens=16,
                 downsample_times=[2, 1],
                 depths = [3, 3],
                 num_heads=[8,16],
                 use_SRM=False,

                 ):

        super(Restormer, self).__init__()
        drop_path_rate = 0.1
        self.dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.use_SRM = use_SRM
        if self.use_SRM:
            print("We are using SRM in Inception")
            ## bayar conv
            self.BayarConv2D_3x3 = nn.Conv2d(3, 3 * self.use_SRM, 3, 1, padding=1, bias=True)
            self.BayarConv2D_5x5 = nn.Conv2d(3, 3 * self.use_SRM, 5, 1, padding=2, bias=True)
            self.BayarConv2D_7x7 = nn.Conv2d(3, 3 * self.use_SRM, 7, 1, padding=3, bias=True)
            self.bayar_mask_3x3 = (torch.tensor(np.ones(shape=(3, 3)))).cuda()
            self.bayar_mask_3x3[1, 1] = 0
            self.bayar_final_3x3 = (torch.tensor(np.zeros((3, 3)))).cuda()
            self.bayar_final_3x3[1, 1] = -1
            self.bayar_mask_5x5 = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
            self.bayar_mask_5x5[2, 2] = 0
            self.bayar_final_5x5 = (torch.tensor(np.zeros((5, 5)))).cuda()
            self.bayar_final_5x5[2, 2] = -1
            self.bayar_mask_7x7 = (torch.tensor(np.ones(shape=(7, 7)))).cuda()
            self.bayar_mask_7x7[3, 3] = 0
            self.bayar_final_7x7 = (torch.tensor(np.zeros((7, 7)))).cuda()
            self.bayar_final_7x7[3, 3] = -1

            self.init_conv = nn.Conv2d(inp_channels, dim-3*3*self.use_SRM, 7, padding=3)
        else:
            self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)



        self.in_chans = inp_channels
        self.out_chans = out_channels
        self.num_heads = num_heads
        self.drop_tokens_per_layer = [
            1. / drop_tokens,
            1. / (drop_tokens / 4),
                                      ]  # default: [1/64,1/16,1/4,1.0]

        self.downsample_times = downsample_times
        self.num_tokens_per_layer = [
            (img_size // (2 ** (downsample_times[0]))) ** 2,
            (img_size // (2 ** (downsample_times[0] + downsample_times[1]))) ** 2,
        ]

        self.num_tokens_after_grouping = [int(self.num_tokens_per_layer[i] * self.drop_tokens_per_layer[i])
                                          for i in range(len(self.num_tokens_per_layer))]

        ## todo: replace the first trans block with convnext block
        # self.encoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type)
        #     # Block(
        #     #     dim=dim, num_heads=heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
        #     #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(num_blocks[0])])
        # self.group1 = GroupingBlock_Prompt(
        #     dim=dim,
        #     out_dim=self.unified_dim,
        #     num_heads=heads[0],
        #     num_group_token=self.num_tokens_after_grouping[i],
        #     num_output_group=self.num_tokens_after_grouping[i],
        # )


        cur = 0
        layer_scale_init_value = 1e-6
        self.encoder_level1 = nn.Sequential(
                *[ConvNextBlock(dim=dim, drop_path=self.dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[0])]
            )
        cur += depths[0]

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        # self.encoder_level2 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type)
        #     # Block(
        #     #     dim=int(dim * 2 ** 1), num_heads=heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
        #     #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, act_layer=act_layer)
        #
        #     for i in range(num_blocks[1])])
        self.encoder_level2 = nn.Sequential(
            *[ConvNextBlock(dim=int(dim * 2 ** 1), drop_path=self.dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[1])]
        )
        cur += depths[1]

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            # Block(
            #     dim=int(dim * 2 ** 2), num_heads=heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, act_layer=act_layer)

            for i in range(num_blocks[2])])

        self.group_embeddings2_3 = nn.Embedding(self.num_tokens_after_grouping[0], int(dim * 2 ** 2))
        self.position_group_tokens2_3 = torch.arange(start=0, end=self.num_tokens_after_grouping[0]).cuda()

        self.group2_3 = GroupingBlock_Prompt(
                dim=int(dim * 2 ** 2),
                out_dim=int(dim * 2 ** 2),
                num_heads=num_heads[0],
                num_group_token=self.num_tokens_after_grouping[0],
                num_output_group=self.num_tokens_after_grouping[0],
                gumbel=False,
                hard=False
            )
        self.encoder_level3_prompt_decoder = nn.ModuleList(
            [Decode_Block(int(dim * 2 ** 2)) for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            # Block(
            #     dim=int(dim * 2 ** 3), num_heads=heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            #     init_values=init_values,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, act_layer=act_layer)

            for i in range(num_blocks[3])])

        self.group_embeddings_latent = nn.Embedding(self.num_tokens_after_grouping[1], int(dim * 2 ** 3))
        self.position_group_tokens_latent = torch.arange(start=0, end=self.num_tokens_after_grouping[1]).cuda()

        self.group_latent = GroupingBlock_Prompt(
            dim=int(dim * 2 ** 3),
            out_dim=int(dim * 2 ** 3),
            num_heads=num_heads[1],
            num_group_token=self.num_tokens_after_grouping[1],
            num_output_group=self.num_tokens_after_grouping[1],
        )
        self.latent_prompt_decoder = nn.ModuleList(
            [Decode_Block(int(dim * 2 ** 3)) for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            # Block(
            #     dim=int(dim * 2 ** 2), num_heads=heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            #     init_values=init_values,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, act_layer=act_layer)

            for i in range(num_blocks[2])])

        self.group_embeddings4_3 = nn.Embedding(self.num_tokens_after_grouping[0], int(dim * 2 ** 2))
        self.position_group_tokens4_3 = torch.arange(start=0, end=self.num_tokens_after_grouping[0]).cuda()

        self.group4_3 = GroupingBlock_Prompt(
            dim=int(dim * 2 ** 2),
            out_dim=int(dim * 2 ** 2),
            num_heads=num_heads[0],
            num_group_token=self.num_tokens_after_grouping[0],
            num_output_group=self.num_tokens_after_grouping[0],
        )
        self.decoder_level3_prompt_decoder = nn.ModuleList(
            [Decode_Block(int(dim * 2 ** 2)) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level2 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type)
        #     # Block(
        #     #     dim=int(dim * 2 ** 1), num_heads=heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #     #     init_values=init_values,
        #     #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, act_layer=act_layer)
        #
        #     for i in range(num_blocks[1])])
        self.decoder_level2 = nn.Sequential(
            *[ConvNextBlock(dim=int(dim * 2 ** 1), drop_path=self.dp_rates[cur - 1 - j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[1])]
        )
        cur -= depths[1]

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        # self.decoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type)
        #     # Block(
        #     #     dim=int(dim * 2 ** 1), num_heads=heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #     #     init_values=init_values,
        #     #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, act_layer=act_layer)
        #
        #     for i in range(num_blocks[0])])
        self.decoder_level1 = nn.Sequential(
            *[ConvNextBlock(dim=int(dim * 2 ** 1), drop_path=self.dp_rates[cur - 1 - j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[0])]
        )
        cur -= depths[0]

        # self.refinement = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type)
        #     # Block(
        #     #     dim=int(dim * 2 ** 1), num_heads=heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #     #     init_values=init_values,
        #     #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, act_layer=act_layer)
        #
        #     for i in range(num_refinement_blocks)])
        self.refinement = nn.Sequential(
            *[ConvNextBlock(dim=int(dim * 2 ** 1), drop_path=0.,
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[0])]
        )
        cur -= depths[0]


        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        B = inp_img.shape[0]
        # inp_enc_level1 = self.patch_embed(inp_img)
        if self.use_SRM:
            x = inp_img
            for i in range(self.use_SRM):
                self.BayarConv2D_3x3.weight.data[3*i:3*(i+1)] *= self.bayar_mask_3x3
                self.BayarConv2D_3x3.weight.data[3*i:3*(i+1)] *= torch.pow(
                    self.BayarConv2D_3x3.weight.data[3*i:3*(i+1)].sum(axis=(2, 3)).view(3, 3, 1, 1),-1)
                self.BayarConv2D_3x3.weight.data[3*i:3*(i+1)] += self.bayar_final_3x3

                self.BayarConv2D_5x5.weight.data[3 * i:3 * (i + 1)] *= self.bayar_mask_5x5
                self.BayarConv2D_5x5.weight.data[3 * i:3 * (i + 1)] *= torch.pow(
                    self.BayarConv2D_5x5.weight.data[3 * i:3 * (i + 1)].sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
                self.BayarConv2D_5x5.weight.data[3 * i:3 * (i + 1)] += self.bayar_final_5x5

                self.BayarConv2D_7x7.weight.data[3 * i:3 * (i + 1)] *= self.bayar_mask_7x7
                self.BayarConv2D_7x7.weight.data[3 * i:3 * (i + 1)] *= torch.pow(
                    self.BayarConv2D_7x7.weight.data[3 * i:3 * (i + 1)].sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
                self.BayarConv2D_7x7.weight.data[3 * i:3 * (i + 1)] += self.bayar_final_7x7
            inp_enc_level1 = torch.cat([self.BayarConv2D_3x3(x), self.BayarConv2D_5x5(x), self.BayarConv2D_7x7(x), self.init_conv(x)], dim=1)
        else:
            inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        ## translayer
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)


        out_enc_level3 = self.adjust_by_prompt(ori_output_feat=out_enc_level3,
                                               position_group_tokens=self.position_group_tokens2_3,
                                               group_embeddings=self.group_embeddings2_3,
                                               grouping_layer=self.group2_3,
                                               encoder_layer=self.encoder_level3,
                                               encoder_prompt_decoder=self.encoder_level3_prompt_decoder,
                                               ori_input_feat=inp_enc_level3,
                                               )

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        latent = self.adjust_by_prompt(ori_output_feat=latent,
                                               position_group_tokens=self.position_group_tokens_latent,
                                               group_embeddings=self.group_embeddings_latent,
                                               grouping_layer=self.group_latent,
                                               encoder_layer=self.latent,
                                               encoder_prompt_decoder=self.latent_prompt_decoder,
                                               ori_input_feat=inp_enc_level4,
                                               )

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        out_dec_level3 = self.adjust_by_prompt(ori_output_feat=out_dec_level3,
                                               position_group_tokens=self.position_group_tokens4_3,
                                               group_embeddings=self.group_embeddings4_3,
                                               grouping_layer=self.group4_3,
                                               encoder_layer=self.decoder_level3,
                                               encoder_prompt_decoder=self.decoder_level3_prompt_decoder,
                                               ori_input_feat=inp_dec_level3,
                                               )

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)  # + inp_img
            if self.in_chans == self.out_chans:
                out_dec_level1 += inp_img

        return out_dec_level1, None

    def adjust_by_prompt(self, *, ori_output_feat, position_group_tokens, group_embeddings, grouping_layer, encoder_layer,
                         encoder_prompt_decoder, ori_input_feat):
        B = ori_output_feat.shape[0]
        out_enc_level3_proj = rearrange(ori_output_feat, 'b c h w -> b (h w) c')
        positions = position_group_tokens.repeat(B, 1)
        group_token = group_embeddings(positions)
        prompt_level3 = grouping_layer(out_enc_level3_proj, group_token)
        td = []
        for idx_prompt_proj, layer in enumerate(encoder_layer):
            prompt_level3, out = encoder_prompt_decoder[idx_prompt_proj](prompt_level3)
            td = [out] + td

        out_enc_level3 = ori_input_feat
        for idx_forward, layer in enumerate(encoder_layer):
            td_x = rearrange(td[idx_forward], 'b (h w) c -> b c h w', h=out_enc_level3.shape[2],
                             w=out_enc_level3.shape[3])
            out_enc_level3 = encoder_layer[idx_forward](out_enc_level3, td=td_x)

        return out_enc_level3


if __name__ == '__main__':
    with torch.no_grad():
        model = Restormer(img_size=256)
        input = torch.ones((1, 3, 256, 256))
        output = model(input)
        print(output.shape)

