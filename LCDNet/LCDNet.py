from functools import partial

import antialiased_cnns
import math
import torch
import torch.nn as nn
from thop import profile, clever_format
from timm.layers import trunc_normal_tf_
from timm.models import named_apply
import torch.nn.functional as F
from torch import Tensor

from AACRNet.AACRNet import kernel_size
from LCDNet.Zoo.EMCAD import _init_weights


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

# from ECANet, in which y and b is set default to 2 and 1
def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class BasicBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.dim = in_channel
        self.dim_learn = self.dim // 2
        self.dim_untouched = self.dim - self.dim_learn
        self.MSCB = MSCB(self.dim_learn, self.dim_learn)

    def forward(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_learn, self.dim_untouched], dim=1)
        x2 = self.MSCB(x2)
        x = torch.cat((x1, x2), 1)
        x = channel_shuffle(x, gcd(self.dim, self.dim_learn))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, groups=in_channel, padding=1,
                                             stride=1, bias=False),
                                   nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0,
                                             bias=False),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU6(inplace=True),
                                   )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.anti = antialiased_cnns.BlurPool(in_channel, stride=2)
        self.fuse = nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=1, stride=1)

    def forward(self, x):
        # 卷积分支
        output1 = self.conv1(x)
        output1 = self.anti(output1)
        # 池化分支
        output2 = self.pool(x)
        output2 = self.anti(output2)
        output = torch.cat([output1, output2], dim=1)
        output = self.fuse(output)
        return output


class LightBackboneV2(nn.Module):
    def __init__(self, net_size=1.0):
        super(LightBackboneV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']
        in_channels = configs[net_size]['in_channels']
        self.stem = nn.Sequential(nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channels),
                                  nn.ReLU6(inplace=True)
                                  )
        self.in_channels = in_channels

        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.layer4 = self._make_layer(out_channels[3], num_blocks[3])

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        x1 = self.layer1(out)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


configs = {
    0.5: {
        'in_channels': 16,
        'out_channels': (32, 64, 128, 256),
        'num_blocks': (1, 1, 1, 1)
    },
    1.0: {
        'in_channels': 16,
        'out_channels': (32, 64, 128, 256),
        'num_blocks': (3, 7, 3, 3)
    },
    2.0: {
        'in_channels': 32,
        'out_channels': (64, 128, 256, 512),
        'num_blocks': (1, 1, 1, 1)
    }
}


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride=1, activation='gelu', dw_parallel=True):
        super(MSDC, self).__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')
        self.gate = ChannelGate(self.in_channels)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            dw_out = self.gate(dw_out) + dw_out
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=None, stride=1, expansion_factor=2, dw_parallel=True,
                 add=True, activation='gelu'):
        super(MSCB, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        self.stride = stride
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)

        self.combined_channels = self.ex_channels * 1

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')
        # self.gate = ChannelGateV2(self.ex_channels)
        self.gate = ChannelGate(self.ex_channels)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
            self,
            in_channels,
            num_gates=None,
            return_gates=False,
            gate_activation='sigmoid',
            reduction=16,
            layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )
        att_kernel = 7
        att_padding = att_kernel // 2
        self.s_att = nn.Conv2d(in_channels, in_channels, 7, 1, att_padding, groups=in_channels,
                               bias=False)
        self.gate_fn = nn.Sigmoid()
    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x_hw = self.s_att(input)
        x = x + x_hw
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x + input
class AlignedModulev2(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignedModulev2, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane, 2, kernel_size=1, padding=0, bias=False)
        self.fusion1 = nn.Sequential(
            nn.Conv2d(outplane + inplane, outplane, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(outplane),
            nn.ReLU6()
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(outplane + inplane, outplane, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(outplane),
            nn.ReLU6()
        )

    def forward(self, l_feature, h_feature):
        h, w = l_feature.size()[2:]
        size = (h, w)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        fuse = self.fusion1(torch.cat([l_feature, h_feature],dim=1))
        flow = self.flow_make(fuse)
        h_feature_warp = self.flow_warp(h_feature, flow, size=size)
        fuse_feature = torch.cat([h_feature_warp, l_feature], dim=1)
        fuse_out = self.fusion2(fuse_feature) + fuse
        return fuse_out

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=True)
        return output


class SizeAware_Decoder(nn.Module):
    """Basic block in SizeAware_Decoder."""
    def __init__(self, in_channel, out_channel, activation='gelu'):
        super().__init__()
        self.activation = activation
        # 小尺度特征卷积（1x1卷积）
        self.small = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, padding=0),  # 点卷积
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, groups=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            act_layer(self.activation, inplace=True)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, padding=0),  # 点卷积
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, groups=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            act_layer(self.activation, inplace=True)
        )
        self.large = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, padding=0),  # 点卷积
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, groups=out_channel, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channel),
            act_layer(self.activation, inplace=True)
        )
        self.gate = ChannelGate(out_channel)
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  act_layer(self.activation, inplace=True)
                                  )
        self.align = AlignedModulev2(in_channel, out_channel)

    def forward(self, high, low):
        output = self.align(low, high)
        small = self.small(output)
        middle = self.middle(output)
        large = self.large(output)
        weight_small = self.gate(small)  # 输出为[batch_size, 1, h, w]
        weight_middle = self.gate(middle)  # 输出为[batch_size, 1, h, w]
        weight_large = self.gate(large)  # 输出为[batch_size, 1, h, w]
        # 自适应尺度感知
        fusion = weight_small + weight_middle + weight_large
        out = self.fuse(fusion) + output
        return out


class TAM(nn.Module):
    def __init__(self, in_channel):
        super(TAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # ECA的部分
        self.k = kernel_size(in_channel)

        self.channel_conv1 = nn.Conv1d(2, 1, kernel_size=self.k, padding=self.k // 2)

        self.spatial_conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, log=False, module_name=None, img_name=None):
        # change part
        diff = torch.abs(t1 - t2)
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1).squeeze(-1).transpose(1, 2)  # b,c,1,1

        t2_channel_avg_pool = self.avg_pool(t2).squeeze(-1).transpose(1, 2)  # b,c,1,1

        diff_channel_avg_pool = self.avg_pool(diff).squeeze(-1).transpose(1, 2)  # b,c,1,1

        channel_pool1 = torch.cat([t1_channel_avg_pool,
                                   diff_channel_avg_pool],
                                  dim=1)  # b,4,c
        channel_pool2 = torch.cat([t2_channel_avg_pool,
                                   diff_channel_avg_pool],
                                  dim=1)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool1).transpose(1, 2).unsqueeze(-1)  # b,1,c
        t2_channel_attention = self.channel_conv1(channel_pool2).transpose(1, 2).unsqueeze(-1)  # b,1,c

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        diff_spatial_avg_pool = torch.mean(diff, dim=1, keepdim=True)  # b,1,h,w

        spatial_pool1 = torch.cat([t1_spatial_avg_pool, diff_spatial_avg_pool], dim=1)  # b,4,h,w
        spatial_pool2 = torch.cat([t2_spatial_avg_pool, diff_spatial_avg_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool1)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv1(spatial_pool2)  # b,1,h,w

        t1_w = self.sigmoid(t1_spatial_attention + t1_channel_attention)
        t2_w = self.sigmoid(t2_spatial_attention + t2_channel_attention)
        fuse1 = t1_w * t1
        fuse2 = t2_w * t2
        fuse = fuse1 + fuse2
        return fuse


class LOSNet(nn.Module):
    def __init__(self):
        super(LOSNet, self).__init__()
        channel_list = [16, 32, 64, 128, 256]
        self.backbone = LightBackboneV2(net_size=0.5)

        self.tam2 = TAM(channel_list[1])
        self.tam3 = TAM(channel_list[2])
        self.tam4 = TAM(channel_list[3])
        self.tam5 = TAM(channel_list[4])

        self.SAD3 = SizeAware_Decoder(channel_list[4], channel_list[3])
        self.SAD2 = SizeAware_Decoder(channel_list[3], channel_list[2])
        self.SAD1 = SizeAware_Decoder(channel_list[2], channel_list[1])

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(channel_list[1], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_change = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, t1, t2, log=False, img_name=None):
        # # 使用 self.backbone 提取多尺度特征
        t1_features = self.backbone(t1)  # 提取 t1 的多尺度特征
        t1_2, t1_3, t1_4, t1_5 = t1_features
        #
        t2_features = self.backbone(t2)  # 提取 t2 的多尺度特征
        t2_2, t2_3, t2_4, t2_5 = t2_features

        d1 = self.tam2(t1_2, t2_2)  # 1,32,128,128
        d2 = self.tam3(t1_3, t2_3)  # 1,64,64,64
        d3 = self.tam4(t1_4, t2_4)  # 1,128,32,32
        d4 = self.tam5(t1_5, t2_5)  # 1,256,16,16

        p3 = self.SAD3(d4, d3)

        p2 = self.SAD2(p3, d2)

        p1 = self.SAD1(p2, d1)

        change_out = self.upsample_x2(p1)
        change_out = self.conv_out_change(change_out)

        return change_out


if __name__ == '__main__':
    a = torch.rand(1, 3, 256, 256)
    b = torch.rand(1, 3, 256, 256)
    model = LOSNet()
    # # 计算模型参数量
    flops, params = profile(model, inputs=(a, b))
    flops, params = clever_format([flops, params], "%.3f")
    print(params)
    print(flops)
    res = model(a, b)
    print(res.shape)
