import math
import torch.nn as nn
from timm.models.layers import trunc_normal_


from lib.model.layers.attn import Attention_gui


class LPU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.conv(x) + x

class CFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        #
        self.conv33 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                                groups=in_channels)
        self.bn33 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0,
                                groups=in_channels)
        self.bn11 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        #
        self.conv_up = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1,
                                 padding=0)
        self.bn_up = nn.BatchNorm2d(in_channels * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()

        self.conv_down = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_down = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # down
        self.adjust = nn.Conv2d(in_channels, out_channels, 1)

        # norm all
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        x=self.bn11(x)
        residual = self.residual(x)


        x = x+self.bn11(self.conv11(x)) + self.bn33(self.conv33(x))

        x = self.conv11(x) + self.bn_down(self.conv_down(self.act(self.conv_up(x))))

        x = self.adjust(x)

        out = self.norm(residual + x)
        return out


class SIFF(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = Attention_Module(dim=dim, reduction=reduction, num_heads=num_heads)
        self.cfn = CFN(in_channels=dim, out_channels=dim)
        self.apply(self._init_weights)

        self.lpu = LPU(dim)

    def forward(self, x1, x2):
        x1 = self.lpu(x1)
        x2 = self.lpu(x2)

        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = x1 + x2
        merge = self.cfn(merge, H, W)

        return merge

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class Attention_Module(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = Attention_gui(dim // reduction, num_heads=num_heads)
        self.end_proj = nn.Linear(dim, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.linear(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.linear(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)

        y1 = y1 + v1
        y2 = y2 + v2

        out_x1 = self.norm1(x1 + self.end_proj(y1))
        out_x2 = self.norm2(x2 + self.end_proj(y2))
        return out_x1, out_x2
