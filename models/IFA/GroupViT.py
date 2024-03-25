import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

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

class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        ## todo: query is group tokens , key is image tokens
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x

from itertools import repeat
import collections.abc
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        ## todo: query [num_query, dim]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        ## todo: k+v [num_image, dim]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        ## todo: attn [num_query, num_image] v [num_image dim]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'

class AssignAttention_Prompt(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key, *, value=None, return_attn=False):
        B, N, C = query.shape
        # if key is None:
        #     key = query
        if value is None:
            value = query # previously key
        S = key.size(1)
        # [B, nh, N, C//nh]
        ## todo: query [num_query, dim]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        ## todo: k+v [num_image, dim]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        # if not self.sum_assign:
        #     attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        # attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        ## todo: attn [num_query, num_image] v [num_image dim]
        mapped = attn.transpose(-2,-1) @ v ## previously attn @ v
        out = rearrange(mapped, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=S, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'



class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 num_output_group,
                 norm_layer=nn.LayerNorm,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.):
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        # [B, S_2, C]
        # projected_group_tokens = self.project_group_token(group_tokens)
        ## todo: project the tokens onto the image
        projected_group_tokens = self.pre_assign_attn(query=group_tokens, key=x)
        ## todo: assign
        new_x, attn_dict = self.assign(query=projected_group_tokens, key=x, return_attn=return_attn)
        new_x += projected_group_tokens

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        return new_x, attn_dict


class GroupingBlock_Prompt(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 num_output_group,
                 norm_layer=nn.LayerNorm,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.):
        super(GroupingBlock_Prompt, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention_Prompt(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        # [B, S_2, C]
        # projected_group_tokens = self.project_group_token(group_tokens)
        ## todo: project the tokens onto the image
        projected_group_tokens = self.pre_assign_attn(query=group_tokens, key=x)
        ## todo: assign
        new_x, attn_dict = self.assign(query=projected_group_tokens, key=x, return_attn=return_attn)
        # new_x += projected_group_tokens

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        return new_x

if __name__ == '__main__':
    x = torch.ones((1,200, 256))
    group = torch.ones((1,20, 256))
    grouping_layer = GroupingBlock_Prompt(
                    dim=256,
                    out_dim=256,
                    num_heads=4,
                    num_group_token=20,
                    num_output_group=20,
                )
    grouped = grouping_layer(x,group)
    print(grouped.shape)