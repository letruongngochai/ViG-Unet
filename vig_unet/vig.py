# from gcn import MRGraphConv2d
# from misc import DropPath2D, Identity, trunc_array

import numpy as np
from torch import nn
import torch.nn.functional as F
from .grapher import Grapher
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# class Grapher(nn.Module):
#     def __init__(self, in_feat, hidden_feat = None, out_feat = None, drop_path = 0., k = 9, dilation = 1):
#         super().__init__()
#         out_feat = out_feat or in_feat
#         hidden_feat = hidden_feat or in_feat
#         self.fc1 = nn.Sequential(
#             nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_feat),
#         )
#         self.graph_conv = nn.Sequential(
#             MRGraphConv2d(in_feat, hidden_feat, k=k, dilation=dilation),
#             nn.BatchNorm2d(hidden_feat),
#             nn.GELU(),
#         )
#         self.fc2 = nn.Sequential(
#             nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_feat),
#         )
#         self.drop_path = DropPath2D(drop_path) if drop_path > 0. else Identity()

#     def forward(self, x):
#         shortcut = x
#         print(x.shape)
#         x = self.fc1(x)
#         x = self.graph_conv(x)
#         x = self.fc2(x)
#         x = shortcut + self.drop_path(x)
#         return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
    
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = shortcut + self.drop_path(x)
        return x
    
class ViG_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0., dilation=1):
        super().__init__()
        self.grapher = Grapher(dim, act='gelu', drop_path=drop_path, dilation=dilation)
        self.mlp = FFN(dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop_path=drop_path)
        
    def forward(self, x):
        x = self.grapher(x)
        x = self.mlp(x)
        return x