import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Attention

class Block(nn.Module):

    def __init__(self, dim, num_heads, n_experts, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = Switch_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              n_experts=n_experts)

    def forward(self, x, indexes_list):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), indexes_list))
        return x

class Switch_Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, n_experts, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                act_layer(),
                nn.Dropout(drop),
                nn.Linear(hidden_features, out_features),
                act_layer(),
            )
            for i in range(n_experts)])
        self.n_experts = n_experts
        self.switch = nn.Linear(in_features, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, routes):
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]
        final_output = x.new_zeros(x.shape)
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]
        final_output = final_output.view(batch_size,seq_len, d_model)

        return final_output
