import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *
from .vig import ViG_Block, Grapher

class ViG_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, k=9, drop_path=0., dilation=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.stem = Stem(n_channels, 32)
        self.grapher_ffn1 = ViG_Block(32, drop_path=drop_path, dilation=dilation)
        
        self.down1 = Down(32, 64)
        self.grapher_ffn2 = ViG_Block(64, drop_path=drop_path, dilation=dilation)
        
        self.down2 = Down(64, 128)
        self.grapher_ffn3 = ViG_Block(128, drop_path=drop_path, dilation=dilation)
        
        self.down3 = Down(128, 256)
        self.grapher_ffn4 = ViG_Block(256, drop_path=drop_path, dilation=dilation)
        
        self.down4 = Down(256, 512)
        self.grapher1 = Grapher(512, drop_path=drop_path, dilation=dilation)
        self.grapher2 = Grapher(512, drop_path=drop_path, dilation=dilation)
        
        self.up1 = Up(512 + 256, 256) 
        self.grapher_ffn7 = ViG_Block(256, drop_path=drop_path, dilation=dilation)
        
        self.up2 = Up(256 + 128, 128)
        self.grapher_ffn8 = ViG_Block(128, drop_path=drop_path, dilation=dilation)
        
        self.up3 = Up(128 + 64, 64)
        self.grapher_ffn9 = ViG_Block(64, drop_path=drop_path, dilation=dilation)
        
        self.up4 = Up(64 + 32, 32)
        self.grapher_ffn10 = ViG_Block(32, drop_path=drop_path, dilation=dilation)
        
        self.out_conv = OutConv(32, n_classes)
        

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.grapher_ffn1(x1)

        x2 = self.down1(x1)
        x2 = self.grapher_ffn2(x2)

        x3 = self.down2(x2)
        x3 = self.grapher_ffn3(x3)

        x4 = self.down3(x3)
        x4 = self.grapher_ffn4(x4)

        x5 = self.down4(x4)
        x5 = self.grapher1(x5)
        x5 = self.grapher2(x5)
        # print("x1", x1.size())
        # print("x2", x2.size())
        # print("x4", x4.size())
        # print("x5", x5.size())

        x = self.up1(x5, x4)
        x = self.grapher_ffn7(x)

        x = self.up2(x, x3)
        x = self.grapher_ffn8(x)

        x = self.up3(x, x2)
        x = self.grapher_ffn9(x)

        x = self.up4(x, x1)
        x = self.grapher_ffn10(x)

        outputs = self.out_conv(x)
        return outputs

# model = ViG_Unet(n_channels=3, n_classes=3)
# print(summary(model, (3, 224, 224))) #572x572
