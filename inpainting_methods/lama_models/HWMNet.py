import torch
import torch.nn as nn
# from .WT import DWT, IWT


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print(x_HH[:, 0, :, :])
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).cuda() #

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt_init(x)


##---------- Basic Layers ----------
def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def bili_resize(factor):
    return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

##---------- Basic Blocks ----------
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, style_control=False, use_dwt=True):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.style_control = style_control
        print(f"enable style control:{style_control}")
        self.use_dwt = use_dwt
        ### previously
        self.body = [HWB(n_feat=in_size, o_feat=in_size, kernel_size=3, reduction=16, bias=False, act=nn.PReLU(), use_dwt=self.use_dwt)]# for _ in range(wab)]
        ### alternatively: FFC ResBlock from LAMA
        # self.body = [FFCResnetBlock(in_size, ratio_gin=0.5, ratio_gout=0.5)]
        self.body = nn.Sequential(*self.body)

        if downsample:
            self.downsample = PS_down(out_size, out_size, downscale=2)

        self.tail = nn.Conv2d(in_size, out_size, kernel_size=1)
        if self.style_control:
            self.instance_norm = nn.InstanceNorm2d(out_size, affine=False)
            self.condition = Conditional_Norm(in_channels=out_size)


    def forward(self, x, style_code=None):
        out = self.body(x)
        out = self.tail(out)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            if self.style_control:
                out = self.condition(out, style_code)
            return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, style_control=False, use_dwt=True):
        super(UNetUpBlock, self).__init__()
        self.up = PS_up(in_size, out_size, upscale=2)
        self.conv_block = UNetConvBlock(in_size, out_size, downsample=False, style_control=style_control, use_dwt=use_dwt)

    def forward(self, x, bridge, style_code=None):
        up = self.up(x)
        out = torch.cat([up, bridge], dim=1)
        out = self.conv_block(out, style_code)
        return out

##---------- Resizing Modules (Pixel(Un)Shuffle) ----------
class PS_down(nn.Module):
    def __init__(self, in_size, out_size, downscale):
        super(PS_down, self).__init__()
        self.UnPS = nn.PixelUnshuffle(downscale)
        self.conv1 = nn.Conv2d((downscale**2) * in_size, out_size, 1, 1, 0)

    def forward(self, x):
        x = self.UnPS(x)  # h/2, w/2, 4*c
        x = self.conv1(x)
        return x

class PS_up(nn.Module):
    def __init__(self, in_size, out_size, upscale):
        super(PS_up, self).__init__()

        self.PS = nn.PixelShuffle(upscale)
        self.conv1 = nn.Conv2d(in_size//(upscale**2), out_size, 1, 1, 0)

    def forward(self, x):
        x = self.PS(x)  # h/2, w/2, 4*c
        x = self.conv1(x)
        return x

from collections import OrderedDict
def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False, subtask=0):
        super(SKFF, self).__init__()
        self.subtask = subtask
        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


##########################################################################
# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y

##########################################################################
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
# Half Wavelet Dual Attention Block (HWB)
class HWB(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size=3, reduction=16, bias=False, act=nn.ELU(), use_dwt=True):
        super(HWB, self).__init__()
        self.use_dwt = use_dwt

        if self.use_dwt:
            self.dwt = DWT()
            self.iwt = IWT()
        else:
            self.fourier_conv = SpectralTransform(in_channels=n_feat//2, out_channels=n_feat//2)

        modules_body = \
            [
                conv(n_feat*2, n_feat, kernel_size, bias=bias),
                act,
                conv(n_feat, n_feat*2, kernel_size, bias=bias)
            ]
        self.body = nn.Sequential(*modules_body)

        self.WSA = SALayer()
        self.WCA = CALayer(n_feat*2, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x

        # Split 2 part
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        ########## Wavelet domain (Dual attention) ############
        if self.use_dwt:
            x_dwt = self.dwt(wavelet_path_in)
            res = self.body(x_dwt)
            branch_sa = self.WSA(res)
            branch_ca = self.WCA(res)
            res = torch.cat([branch_sa, branch_ca], dim=1)
            res = self.conv1x1(res) + x_dwt
            wavelet_path = self.iwt(res)
        ########## alternatively: fourier path ##########
        else:
            wavelet_path = self.fourier_conv(wavelet_path_in)

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.activate(self.conv3x3(out))
        out += self.conv1x1_final(residual)

        return out


class Conditional_Norm(nn.Module):
    def __init__(self, in_channels=64):
        super(Conditional_Norm, self).__init__()
        # out_channels = in_channels

        # self.res = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), True)
        # conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

        # self.conv_sn_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        # self.act = nn.ELU(inplace=True)
        # self.conv_sn_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)

        self.shared = sequential(torch.nn.Linear(1, in_channels), nn.ReLU())
        self.to_gamma_1 = sequential(torch.nn.Linear(in_channels, in_channels), nn.Sigmoid())
        self.to_beta_1 = sequential(torch.nn.Linear(in_channels, in_channels), nn.Tanh())
        # self.to_gamma_2 = sequential(torch.nn.Linear(in_channels, in_channels), nn.Sigmoid())
        # self.to_beta_2 = sequential(torch.nn.Linear(in_channels, in_channels), nn.Tanh())

    def forward(self, x, label):
        # originally nn.Conv2d((2 ** i) * wf, wf, 3, 1, 1)
        actv = self.shared(label)
        gamma_1, beta_1 = self.to_gamma_1(actv).unsqueeze(-1).unsqueeze(-1), self.to_beta_1(actv).unsqueeze(-1).unsqueeze(-1)
        # gamma_2, beta_2 = self.to_gamma_2(actv).unsqueeze(-1).unsqueeze(-1), self.to_beta_2(actv).unsqueeze(-1).unsqueeze(-1)

        x_1 = gamma_1 * x + beta_1
        # x_2 = self.act(gamma_2 * self.conv_sn_1(x_1) + beta_2)
        return x_1


##########################################################################
##---------- HWMNet-LOL ----------
class HWMNet(nn.Module):
    def __init__(self, in_chn=3, out_chn=None, wf=64, depth=4, subtask=0, style_control=False, use_dwt=True, use_norm_conv=False):
        super(HWMNet, self).__init__()
        self.use_norm_conv = use_norm_conv
        self.subtask = subtask
        self.style_control = style_control
        if out_chn is None:
            out_chn = in_chn
        self.apply_res = in_chn==out_chn
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.bili_down = bili_resize(0.5)
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        # encoder of UNet-64
        prev_channels = 0
        for i in range(depth):  # 0,1,2,3
            downsample = True if (i + 1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels + wf, (2 ** i) * wf, downsample))
            prev_channels = (2 ** i) * wf

        # decoder of UNet-64
        self.up_path = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.bottom_conv = nn.Conv2d(prev_channels, wf, 3, 1, 1)
        self.bottom_up = bili_resize(2 ** (depth-1))

        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2 ** i) * wf, style_control=style_control, use_dwt=use_dwt))
            self.skip_conv.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            self.conv_up.append(nn.Sequential(*[bili_resize(2 ** i), nn.Conv2d((2 ** i) * wf, wf, 3, 1, 1)])) # originally nn.Conv2d((2 ** i) * wf, wf, 3, 1, 1)
            prev_channels = (2 ** i) * wf

        self.final_ff = SKFF(in_channels=wf, height=depth)
        self.last = conv3x3(prev_channels, out_chn, bias=True)

        if self.subtask!=0:
            self.mlp_subtask = sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(wf, wf),
                nn.ReLU(),
                torch.nn.Linear(wf, self.subtask),
                # nn.Sigmoid()
            )

        ## we refine the mask using reparameterization
        if self.use_norm_conv:
            self.global_pool = sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(wf, wf),
                nn.ReLU(),
                # nn.Sigmoid()
            )
            self.to_gamma_1 = sequential(torch.nn.Linear(wf, 1), nn.Sigmoid())
            self.to_beta_1 = sequential(torch.nn.Linear(wf, 1), nn.Tanh())

            self.IN = nn.InstanceNorm2d(1, affine=False)
            self.post_process = nn.Sequential(
                nn.Conv2d(1,16, kernel_size=7, padding=3, dilation=1),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=7, padding=3, dilation=1),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=7, padding=3, dilation=1),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=7, padding=3, dilation=1),
                nn.ELU(),
                nn.Conv2d(16, 1, kernel_size=7, padding=3, dilation=1),
            )

            self.post_process_small_kernel = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, dilation=1),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1),
                nn.ELU(),
                nn.Conv2d(16, 1, kernel_size=7, padding=3, dilation=1),
            )

    def forward(self, x, style_code=None):
        img = x
        scale_img = img

        ##### shallow conv #####
        x1 = self.conv_01(img)
        encs = []
        ######## UNet-64 ########
        # Down-path (Encoder)
        for i, down in enumerate(self.down_path):
            if i == 0:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            elif (i + 1) < self.depth:
                scale_img = self.bili_down(scale_img)
                left_bar = self.conv_01(scale_img)
                x1 = torch.cat([x1, left_bar], dim=1)
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                scale_img = self.bili_down(scale_img)
                left_bar = self.conv_01(scale_img)
                x1 = torch.cat([x1, left_bar], dim=1)
                x1 = down(x1)

        # Up-path (Decoder)
        ms_result = [self.bottom_up(self.bottom_conv(x1))]
        for i, up in enumerate(self.up_path):
            ## up contains upsampling, concat(UNet) and HWAblock
            x1 = up(x1, bridge=self.skip_conv[i](encs[-i - 1]), style_code=style_code)
            ## conv_up contains upsampling and conv for SKFF
            ## Thus we modify the HWA block
            ms_result.append(self.conv_up[i](x1))
        # Multi-scale selective feature fusion
        msff_result = self.final_ff(ms_result)

        ##### Reconstruct #####
        if self.apply_res:
            out_1 = self.last(msff_result) + img
        else:
            out_1 = self.last(msff_result)


        #### sub-task ########
        # if self.subtask != 0:
        #     pred = self.mlp_subtask(msff_result)
        #     if self.use_norm_conv:
        #         out_post = self.post_process(torch.sigmoid(out_1.detach()))
        #         return out_1, pred, out_post
        #     else:
        #         return out_1, pred
        # else:
        if self.use_norm_conv:
            ## minimize the affect on the CE prediction using detach
            # norm_pred = self.IN(torch.sigmoid(out_1.detach()))
            sigmoid_pred = torch.sigmoid(out_1.detach())
            std, mean = torch.std_mean(sigmoid_pred,dim=(2,3))
            norm_pred = (sigmoid_pred-mean.unsqueeze(-1).unsqueeze(-1))/std.unsqueeze(-1).unsqueeze(-1)


            ## get mean and std from msff_result
            actv = self.global_pool(msff_result.detach())
            std_new, mean_new = self.to_gamma_1(actv), self.to_beta_1(actv)
            ## ada instance norm
            adapt_pred = std_new.unsqueeze(-1).unsqueeze(-1) * norm_pred + mean_new.unsqueeze(-1).unsqueeze(-1)

            ## post-process the mask
            diff_pred = self.post_process(adapt_pred)
            post_pred = adapt_pred + diff_pred

            # print(f"original mean/std {mean} {std}")
            # print(f"learned mean/std {mean_new} {std_new}")
            # std_debug, mean_debug = torch.std_mean(adaptive_pred, dim=(2, 3))
            # print(f"debug mean/std {mean_debug} {std_debug}")
            # std_diff, mean_diff = torch.std_mean(diff_pred, dim=(2, 3))
            # print(f"diff mean/std {mean_diff} {std_diff}")
            return out_1, (post_pred, std_new, mean_new), (norm_pred, adapt_pred, diff_pred)
        else:
            return out_1


import torch.nn.functional as F
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


## 核心
class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU()

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


## Spectral Transform
class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

## Fast Fourier Convolution
class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        ### let's suppose in_channel=out_channel=64, and ratio_gin=ratio_gout=0.5

        in_cg = int(in_channels * ratio_gin)   # 32
        in_cl = in_channels - in_cg   # 32
        out_cg = int(out_channels * ratio_gout) # 32
        out_cl = out_channels - out_cg  # 32
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg   # 32

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        ######## BN + ACT ########################
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=True, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)

        self.inline = inline

    def forward(self, x):
        if self.inline:
            # 把通道切分为Local和Global （figure 2）
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

if __name__ == "__main__":
    input = torch.ones(1, 16, 32, 32, dtype=torch.float, requires_grad=False).cuda()

    model = SpectralTransform(in_channels=16, out_channels=16).cuda()
    out = model(input)

    # RDBlayer = SK_RDB(in_channels=64, growth_rate=64, num_layers=3)
    # print(RDBlayer)
    # out = RDBlayer(input)
    # flops, params = profile(RDBlayer, inputs=(input,))
    print('input shape:', input.shape)
    print('output shape', out.shape)




