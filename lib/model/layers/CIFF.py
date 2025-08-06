import torch
import torch.nn as nn
from timm.models.layers import Mlp
from torch.nn import init
import torch.nn.functional as F

from lib.utils.token_utils import patch2token, token2patch

class CB11(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

        # Initialize pwconv layer with Kaiming initialization
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2).contiguous()


class DWC(nn.Module):
    def __init__(self, dim, kernel, padding):
        super().__init__()
        # self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding=padding, groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class LSA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CB11(c2)
        self.dwconv3 = DWC(c2, 3, 1)
        self.dwconv5 = DWC(c2, 5, 2)
        self.dwconv7 = DWC(c2, 7, 3)
        self.pwconv2 = CB11(c2)
        self.fc2 = nn.Linear(c2, c1)

        # Initialize fc1 layer with Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))



class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # Initialize linear layers with Kaiming initialization
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x * y.expand_as(x)).flatten(2).transpose(1, 2).contiguous()



class CIFF_woGIM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.channels = dim

        self.linear = nn.Linear(dim * 2, self.channels)

        self.lsa = LSA(self.channels, self.channels)

        self.se = SE(self.channels)

        # Initialize linear fusion layers with Kaiming initialization
        init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        x = x.flatten(2).transpose(1, 2).contiguous()

        x_sum = self.linear(x)
        x_sum = self.lsa(x_sum, H, W) + self.se(x_sum, H, W)
        x_fusion = x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x1 + x_fusion, x2 + x_fusion

class GIM(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.mlp1 = Mlp(in_features=dim, hidden_features=dim * 2, act_layer=act_layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_v, x_i):
        B, C, H, W = x_v.shape
        N = H * W

        x_v = patch2token(x_v)
        x_i = patch2token(x_i)
        x = torch.cat((x_v, x_i), dim=1)

        x = x + self.norm(self.mlp1(x))
        x_v, x_i = torch.split(x, (N, N,), dim=1)
        x_v = token2patch(x_v)
        x_i = token2patch(x_i)

        return x_v, x_i