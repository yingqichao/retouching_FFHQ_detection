import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
import random
import os

from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class UNetDiscriminator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=3, residual_blocks=8, init_weights=True, use_spectral_norm=True,
                 dim=32, use_sigmoid=False):
        super(UNetDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        # self.clock = 1
        self.in_channels = in_channels

        self.init_conv = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=self.in_channels, out_channels=dim, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )


        self.encoder_1 = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        self.encoder_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 8, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 8, out_channels=dim * 8, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True)
        )

        blocks = []
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 8, dilation=2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(dim * 8, dim * 8)

        self.decoder_3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 8 * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 4 * 2, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2, out_channels=dim * 2,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )

        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2 * 2, out_channels=dim, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),

            spectral_norm(nn.ConvTranspose2d(in_channels=dim, out_channels=dim,  kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.ELU(inplace=True),
        )


        self.decoder_0 = nn.Sequential(
            # spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1),
            #               use_spectral_norm),
            # nn.ELU(),
            # nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=1, padding=0),
        )


        if init_weights:
            self.init_weights()


    def forward(self, x):

        e0 = self.init_conv(x)
        e1 = self.encoder_1(e0)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)

        m = self.middle(e3)
        middle_feat = self.avgpool(m)
        middle_feat = middle_feat.reshape(middle_feat.shape[0], -1)

        d3 = self.decoder_3(torch.cat((e3, m), dim=1))
        d2 = self.decoder_2(torch.cat((e2, d3), dim=1))
        d1 = self.decoder_1(torch.cat((e1, d2), dim=1))
        x = self.decoder_0(torch.cat((e0, d1), dim=1))
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x, middle_feat

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        if use_spectral_norm:
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
                # nn.InstanceNorm2d(dim),
                nn.GELU(),

                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
                # nn.InstanceNorm2d(dim),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,bias=False),
                # nn.BatchNorm2d(dim, affine=True),
                nn.InstanceNorm2d(dim),
                nn.GELU(),

                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,bias=True),
            )


    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) # 224*224
        patch_height, patch_width = pair(patch_size) # 16*16，没有涉及通道

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

if __name__ == '__main__':
    model = UNetDiscriminator()
    input = torch.ones((1,3,128,128))
    output, _ = model(input)
    print(output.shape)